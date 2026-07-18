"""
Multi-survey scanning engine.

A Survey adapter supplies everything schema-specific:
  - connection      : open_cursor() / close_cursor(conn, cur)
  - query shape     : base_from (FROM ... JOIN ...), alias, allowed_sort
  - data shape      : bands, field_map (db col -> template key)
  - per-candidate   : fetch_lightcurve / fetch_limits / fetch_name / fetch_cutouts
  - scans           : name -> {template, builder(form) -> (where_sql, params)}

The generic engine (run_candidate_query, serialize_candidates, render_candidate,
build_scan_context) operates purely on a Survey instance, so adding a new database
means writing one adapter -- no engine changes.

Register new surveys at the bottom in SURVEYS.
"""
import os
import io
import gzip
import logging
import time
import threading
import requests
from dataclasses import dataclass, field as dc_field
from typing import Callable, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats

THUMBDIR = os.environ.get('WTP_THUMBNAIL_DIR',
	'/home/kde/Packages/flask_app/static/thumbnails/')

# ----------------------------------------------------------------------------
# Core types
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Band:
	id: int
	label: str
	color: str

@dataclass
class Survey:
	key: str				# url-safe identifier, e.g. 'wtp', 'prime'
	label: str				# human label
	open_cursor: Callable			# () -> (conn, cur) with a dict cursor
	close_cursor: Callable			# (conn, cur) -> None
	base_from: str				# e.g. 'candidates cand INNER JOIN fields f ON f.field = cand.field'
	alias: str				# candidate table alias, e.g. 'cand'
	id_col: str				# primary id column name, e.g. 'candid'
	bands: list				# list[Band]
	field_map: list				# list[(template_key, db_col)]; must include ('candids', id), ('ras', ra), ('decs', dec)
	allowed_sort: set			# whitelisted ORDER BY columns
	default_sort: tuple			# (col, 'DESC')
	scans: dict				# name -> {'template': str, 'builder': callable}
	fetch_lightcurve: Callable		# (cur, cand) -> {'mjd','mag','magerr','bandid'} as np arrays
	fetch_name: Callable			# (cur, cand) -> str
	fetch_limits: Optional[Callable] = None		# (cur, cand) -> {'mjd','limmag','bandid'} or None
	fetch_cutouts: Optional[Callable] = None	# (cur, cand) -> {'sci','ref','diff'} arrays or None
	fetch_forced: Optional[Callable] = None		# (cur, cand) -> forced-photometry dict or None
	fetch_saved_names: Optional[Callable] = None	# () -> set of Fritz-saved names; enables fr_ex
	cmradius: float = 6.0 / 3600
	name_join: Optional[str] = None		# LATERAL exposing sn.name; enables batched name lookup

SURVEYS = {}					# populated at bottom

# ----------------------------------------------------------------------------
# Fritz saved-source lookup (replaces fritz_source_list.txt)
#
# fr_ex is driven by GET /api/sources?group_ids=<group>, paginated at 500 per
# page, cached per process for FRITZ_SAVED_TTL seconds. On any request failure
# the previous (possibly stale) set is returned -- a scan render must never
# fail because Fritz is down. fritz_mark_saved() seeds a freshly saved source
# into the cache so the badge updates without waiting for the next refresh.
# ----------------------------------------------------------------------------
FRITZ_BASEURL = 'https://fritz.science/'
FRITZ_GROUP_ID = 408
FRITZ_SAVED_TTL = 300
_FRITZ_SAVED = {'names': set(), 'ts': 0.0}
_FRITZ_LOCK = threading.Lock()

def fritz_mark_saved(name):
	with _FRITZ_LOCK:
		_FRITZ_SAVED['names'].add(name)

def wtp_fritz_saved_names():
	if time.time() - _FRITZ_SAVED['ts'] < FRITZ_SAVED_TTL:
		return _FRITZ_SAVED['names']
	token = os.environ.get('FRITZ_TOKEN', '')
	if not token:
		return _FRITZ_SAVED['names']
	with _FRITZ_LOCK:
		if time.time() - _FRITZ_SAVED['ts'] < FRITZ_SAVED_TTL:
			return _FRITZ_SAVED['names']	# refreshed while we waited
		names = set()
		page = 1
		try:
			while True:
				r = requests.get(FRITZ_BASEURL + 'api/sources',
					params={'group_ids': str(FRITZ_GROUP_ID),
					'numPerPage': 500, 'pageNumber': page},
					headers={'Authorization': 'token %s' % token},
					timeout=30)
				data = r.json()['data']
				srcs = data.get('sources', [])
				names.update(s['id'] for s in srcs)
				if len(srcs) < 500:
					break
				page += 1
		except Exception:
			return _FRITZ_SAVED['names']	# stale beats broken
		_FRITZ_SAVED['names'] = names
		_FRITZ_SAVED['ts'] = time.time()
		return names

# ----------------------------------------------------------------------------
# Generic query + safety
# ----------------------------------------------------------------------------
def safe_order(survey, form):
	col = form.get('sb', survey.default_sort[0])
	direction = form.get('so', survey.default_sort[1]).upper()
	if col not in survey.allowed_sort or direction not in ('ASC', 'DESC'):
		raise ValueError('invalid sort specification')
	return col, direction

def run_candidate_query(survey, where_sql, where_params, form,
		extra_join=None, extra_join_params=None, extra_select=None):
	col, direction = safe_order(survey, form)
	skip = max(0, int(float(form.get('skipnum', 0))))
	candlim = max(1, min(int(form.get('candlim', 200)), 1000))
	# Select the serializer's columns (qualified), plus any scan-specific extras.
	cols = ['%s.%s' % (survey.alias, c) for _, c in survey.field_map]
	if extra_select:
		cols += ['%s AS %s' % (expr, key) for key, expr in extra_select]
	# name_join precedes extra_join so a scan's extra LATERAL can reference the
	# source the name lookup already resolved (e.g. sn.wtpid for the NED match),
	# turning that membership join into an indexed wtpid lookup -- no 2nd cone.
	from_clause = survey.base_from
	if survey.name_join:
		cols.append('sn.name AS names_raw')
		from_clause += ' ' + survey.name_join
	if extra_join:
		from_clause += ' ' + extra_join
	if survey.name_join:
		# Dedup by source name in SQL so each page holds exactly candlim
		# unique sources and OFFSET counts deduped sources. Unnamed rows key
		# on candid::text so they never collapse together. The inner ORDER BY
		# picks, per source, the row that is extreme in the user's sort.
		key = 'COALESCE(sn.name, %s.%s::text)' % (survey.alias, survey.id_col)
		inner = ('SELECT DISTINCT ON (%s) %s FROM %s WHERE %s ORDER BY %s, %s.%s %s'
			% (key, ', '.join(cols), from_clause, where_sql, key,
			survey.alias, col, direction))
		sql = ('SELECT * FROM (%s) dq ORDER BY %s %s OFFSET %%s LIMIT %%s'
			% (inner, col, direction))
	else:
		sql = ('SELECT %s FROM %s WHERE %s ORDER BY %s %s OFFSET %%s LIMIT %%s'
			% (', '.join(cols), from_clause, where_sql, col, direction))
	conn, cur = survey.open_cursor()
	try:
		allparams = (list(extra_join_params or []) + list(where_params)
			+ [skip, candlim])
		qstr = cur.mogrify(sql, allparams).decode()
		logging.getLogger(__name__).info(qstr)
		cur.execute(sql, allparams)
		return cur.fetchall(), qstr
	finally:
		survey.close_cursor(conn, cur)

# ----------------------------------------------------------------------------
# Band-agnostic plotting + per-candidate worker
# ----------------------------------------------------------------------------
def _build_figure(survey, cand, lc, lims, name, cut, fp=None):
	"""Build and return the matplotlib Figure. Caller is responsible for closing."""
	candid = cand[survey.id_col]
	if cut is not None:
		fig = Figure(figsize=(10, 10))
		img_axes = [fig.add_subplot(2, 3, i + 1) for i in range(3)]
		ax = fig.add_subplot(2, 1, 2)
		for a, img, ttl in zip(img_axes, (cut['sci'], cut['ref'], cut['diff']),
				('Science', 'Reference', 'Difference')):
			_, med, std = sigma_clipped_stats(img)
			a.imshow(img, cmap='gray', vmin=med - std, vmax=med + 5 * std)
			a.set_title(ttl, fontsize=20)
			a.set_xticks([]); a.set_yticks([])
	else:
		fig = Figure(figsize=(10, 5))
		ax = fig.add_subplot(1, 1, 1)

	# Forced photometry is the primary curve (solid); candidate-based points
	# are faded behind it. Set cand_alpha = 0.3 unconditionally to keep them
	# faded even when no forced photometry exists for the source.
	has_fp = bool(fp and len(fp['mjd']))
	cand_alpha = 0.3 if has_fp else 1.0

	mjd, mag, magerr, bid = lc['mjd'], lc['mag'], lc['magerr'], lc['bandid']
	if lims:
		lmjd = np.asarray(lims['mjd']); lmag = np.asarray(lims['limmag'])
		lbid = np.asarray(lims['bandid'])
	else:
		lmjd = lmag = lbid = np.array([])
	if has_fp:
		fmjd = np.asarray(fp['mjd']); fbid = np.asarray(fp['bandid'])
		fmag = np.asarray(fp['mag']); fmagerr = np.asarray(fp['magerr'])
		flim = np.asarray(fp['limmag']); fdet = np.asarray(fp['isdet'], dtype=bool)
		fsmag = np.asarray(fp.get('smag', [])); fsmagerr = np.asarray(fp.get('smagerr', []))
		fsdet = np.asarray(fp.get('sisdet', []), dtype=bool)
	else:
		fmjd = fbid = fmag = fmagerr = flim = fdet = np.array([])
		fsmag = fsmagerr = np.array([]); fsdet = np.array([], dtype=bool)
	for b in survey.bands:
		# candidate-based detections -- faded behind the forced curve
		d = (bid == b.id)
		if np.any(d):
			ax.errorbar(mjd[d], mag[d], yerr=np.abs(magerr[d]), ls='none',
				marker='s', color=b.color, ms=10, alpha=cand_alpha,
				label=(None if has_fp else b.label))
		# candidate-based upper limits -- faded
		if len(lmjd):
			l = (lbid == b.id)
			if np.any(l):
				ax.errorbar(lmjd[l], lmag[l], yerr=0.2, uplims=True, ls='none',
					marker='s', color=b.color, markerfacecolor='none', ms=8,
					alpha=cand_alpha)
		# forced photometry -- solid detections + solid upper-limit arrows
		if has_fp:
			fb = (fbid == b.id)
			det = fb & fdet & np.isfinite(fmag)
			lim = fb & (~fdet) & np.isfinite(flim)
			if np.any(det):
				ax.errorbar(fmjd[det], fmag[det], yerr=np.abs(fmagerr[det]),
					ls='none', marker='o', color=b.color, ms=11, label=b.label)
			if np.any(lim):
				ax.errorbar(fmjd[lim], flim[lim], yerr=0.2, uplims=True, ls='none',
					marker='v', color=b.color, markerfacecolor='none', ms=8)
			# forced STACK photometry -- open diamonds + thin line, no limits.
			# Tracks the diff curve for hostless sources (ref flux ~ 0); fans
			# brighter than diff for hosted/nuclear except at peak.
			if len(fsmag):
				sb = fb & fsdet & np.isfinite(fsmag)
				if np.any(sb):
					ax.errorbar(fmjd[sb], fsmag[sb], yerr=np.abs(fsmagerr[sb]),
						ls='-', lw=0.8, marker='D', mfc='none', color=b.color, ms=8)
	# Candidate-epoch marker is drawn once below (blue dashed); the old
	# dotted 'scan epoch' line duplicated it at the same x.
	ax.set_xlabel('MJD', fontsize=20)
	ax.set_ylabel('Magnitude', fontsize=20)
	cutcandid = cut.get('candid') if cut is not None else None
	if cutcandid is not None and cutcandid != candid:
		ttl = 'Candidate %d; cutout %d; %s' % (candid, cutcandid, name)
	else:
		ttl = 'Candidate %d; %s' % (candid, name)
	ax.set_title(ttl, fontsize=20)
	ax.tick_params(which='both', labelsize=18)
	ax.invert_yaxis()
	# epoch markers: candidate epoch (blue) and cutout epoch (red).
	_cmjd = cand.get('mjd')
	if _cmjd is not None and np.isfinite(float(_cmjd)):
	    ax.axvline(float(_cmjd), color='b', ls='--', lw=1.0, alpha=0.7)
	_xmjd = cut.get('mjd') if cut is not None else None
	if _xmjd is not None and np.isfinite(float(_xmjd)):
	    ax.axvline(float(_xmjd), color='r', ls='--', lw=1.0, alpha=0.7)
	
	handles, labels = ax.get_legend_handles_labels()
	if handles:
		if has_fp and len(fsmag) and fsdet.any():
			from matplotlib.lines import Line2D
			handles = handles + [
				Line2D([], [], color='0.3', marker='o', ls='none', ms=9),
				Line2D([], [], color='0.3', marker='D', mfc='none', ls='none', ms=8)]
			labels = labels + ['diff forced', 'stack forced']
		ax.legend(handles, labels, fontsize=12)
	if len(mjd) or len(lmjd) or len(fmjd):
		ax2 = ax.twiny()
		ax2.set_xlim(Time(ax.get_xlim(), format='mjd').decimalyear)
		ax2.set_xlabel('Year', fontsize=20)
		ax2.tick_params(labelsize=18)
	fig.tight_layout()
	return fig

def candidate_png_by_id(survey, candid):
	"""Render a candidate's light-curve + cutout figure straight to PNG bytes.

	No disk I/O: served in-memory by the /<survey>/cutout/<candid>.png route.
	"""
	conn, cur = survey.open_cursor()
	try:
		cur.execute('SELECT %s.* FROM %s WHERE %s.%s = %%s LIMIT 1'
			% (survey.alias, survey.base_from, survey.alias, survey.id_col),
			(candid,))
		cand = cur.fetchone()
		if cand is None:
			raise KeyError(candid)
		lc = survey.fetch_lightcurve(cur, cand)
		lims = survey.fetch_limits(cur, cand) if survey.fetch_limits else None
		name = survey.fetch_name(cur, cand)
		cut = survey.fetch_cutouts(cur, cand) if survey.fetch_cutouts else None
		fp = survey.fetch_forced(cur, cand) if getattr(survey, 'fetch_forced', None) else None
		fig = _build_figure(survey, cand, lc, lims, name, cut, fp)
		FigureCanvasAgg(fig)	# private Agg canvas: no global state, no lock
		buf = io.BytesIO()
		fig.savefig(buf, format='png', dpi=80)
		return buf.getvalue()
	finally:
		survey.close_cursor(conn, cur)

# ----------------------------------------------------------------------------
# Serialization (parallel render, dedup by name, build template dict)
# ----------------------------------------------------------------------------
def _load_fritz(survey):
	if survey.fetch_saved_names is None:
		return None
	try:
		return survey.fetch_saved_names()
	except Exception:
		return set()

def serialize_candidates(survey, out):
	keys = [k for k, _ in survey.field_map]
	if not out:
		d = {k: np.array([]) for k in keys}
		d.update({'names': np.array([]), 'gallats': np.array([]),
			'gallongs': np.array([]), 'cutouts': []})
		if survey.fetch_saved_names is not None:
			d['fr_ex'] = np.array([], dtype=bool)
		return d

	d = {k: np.array([o[col] for o in out]) for k, col in survey.field_map}
	# Surface extra_select columns a scan attached (NED/SNX crossmatch:
	# ned_objnames, cludists, proj_sep_kpcs, ned_distmpcs, ned_zs) that are
	# not in field_map, keyed by their SQL alias. dtype=object keeps text/None.
	_used = {col for _, col in survey.field_map}
	for _ek in out[0].keys():
		if _ek not in _used and _ek != 'names_raw' and _ek not in d:
			d[_ek] = np.array([o[_ek] for o in out], dtype=object)
	# Rows carrying names_raw arrive pre-deduplicated by the DISTINCT ON in
	# run_candidate_query -- use them as-is. Custom queries (name/spatial
	# search, YSO scan) fall back to per-candidate fetch_name + Python dedup;
	# 'NULL' (unnamed) rows are exempt so they never collapse into one card.
	if 'names_raw' in out[0]:
		names = np.array([o['names_raw'] if o['names_raw'] else 'NULL'
			for o in out])
	else:
		conn, cur = survey.open_cursor()
		try:
			raw_names = np.array([survey.fetch_name(cur, dict(o)) for o in out])
		finally:
			survey.close_cursor(conn, cur)
		seen, keep = set(), []
		for j, nm in enumerate(raw_names):
			if nm == 'NULL' or nm not in seen:
				seen.add(nm)
				keep.append(j)
		keep = np.array(keep, dtype=int)
		for k in list(d):
			d[k] = d[k][keep]
		names = raw_names[keep]

	coords = SkyCoord(ra=d['ras'], dec=d['decs'], unit='degree', frame='icrs')
	d['names'] = names
	d['gallats'] = coords.galactic.b.degree
	d['gallongs'] = coords.galactic.l.degree
	# Each entry points at the on-demand render route -- no file is written.
	d['cutouts'] = ['/%s/cutout/%d.png' % (survey.key, c) for c in d['candids']]
	saved = _load_fritz(survey)
	if saved is not None:
		d['fr_ex'] = np.array([n in saved for n in names], dtype=bool)
	return d

def ascii_name_list(canddicts):
	"""Plain-text 'name  ra  dec' listing of the returned sources (dedup
	already applied), for copy-paste cross-matching against an existing list."""
	names = list(canddicts.get('names', []))
	ras = list(canddicts.get('ras', []))
	decs = list(canddicts.get('decs', []))
	lines = ['# name              ra           dec']
	for n, r, d in zip(names, ras, decs):
		lines.append('%-18s %11.6f %+11.6f' % (str(n), float(r), float(d)))
	return '\n'.join(lines)


def build_scan_context(survey, scan_name, form):
	"""Run a scan and return (template, context_dict) for render_template."""
	spec = survey.scans.get(scan_name)
	if spec is None:
		raise KeyError('unknown scan %r for survey %r' % (scan_name, survey.key))
	where_sql, params = spec['builder'](form)
	join, join_params = spec.get('join'), []
	if callable(join):
		join, join_params = join(form)
	out, qstr = run_candidate_query(survey, where_sql, params, form, extra_join=join,
		extra_join_params=join_params, extra_select=spec.get('extra'))
	reqcopy = form.to_dict()
	reqcopy['skipnum'] = int(float(form.get('skipnum', 0))) + len(out)
	canddicts = serialize_candidates(survey, out)
	ctx = dict(numcands=len(canddicts['candids']), canddicts=canddicts, defpardict=reqcopy,
		scan_title=spec.get('title', scan_name),
		scan_fields=spec.get('fields', []),
		scan_query=qstr,
		scan_action='/%s/scan/%s' % (survey.key, scan_name),
		ascii_list=ascii_name_list(canddicts))
	return spec['template'], ctx

# ----------------------------------------------------------------------------
# Connection pooling
#
# One ThreadedConnectionPool per logical database, created lazily inside each
# process. Do NOT build pools at import time -- gunicorn --preload would then
# share sockets across forked workers. _pool() builds on first use per process.
# ----------------------------------------------------------------------------
import threading
import psycopg2
import psycopg2.pool
import psycopg2.extras

_POOLS = {}
_POOL_LOCK = threading.Lock()

def make_pool_adapter(name, dsn_env, minconn=1, maxconn=8):
	"""Return (open_cursor, close_cursor) backed by a lazy ThreadedConnectionPool.

	Connections are autocommit (scanning is read-only) and rolled back on return,
	so an aborted transaction never poisons the next borrower.
	"""
	def _pool():
		pool = _POOLS.get(name)
		if pool is None:
			with _POOL_LOCK:
				pool = _POOLS.get(name)
				if pool is None:
					pool = psycopg2.pool.ThreadedConnectionPool(
						minconn, maxconn, os.environ[dsn_env])
					_POOLS[name] = pool
		return pool

	def open_cursor():
		conn = _pool().getconn()
		conn.autocommit = True
		cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
		return conn, cur

	def close_cursor(conn, cur):
		try:
			cur.close()
		except Exception:
			pass
		try:
			conn.rollback()
		except Exception:
			pass
		_pool().putconn(conn)

	return open_cursor, close_cursor

# ============================================================================
# ADAPTER 1: WTP (unWISE / CatWISE2020, W1/W2)
# ============================================================================
# WTP uses the pooled adapter. WTP_DSN is required (lazy: read on first
# connection, not at import); WTP_POOL_MAX optionally sizes the pool.
_wtp_open, _wtp_close = make_pool_adapter('wtp', 'WTP_DSN',
	maxconn=int(os.environ.get('WTP_POOL_MAX', '8')))

WTP_CMRADIUS = 5.0 / 3600

def wtp_fetch_lightcurve(cur, cand):
	cur.execute('SELECT mjd, psf_mag, psf_mag_err, bandid FROM candidates '
		'WHERE q3c_radial_query(ra, dec, %s, %s, %s) AND ispos = 1 ORDER BY mjd DESC;',
		(cand['ra'], cand['dec'], WTP_CMRADIUS))
	out = cur.fetchall()
	return {'mjd': np.array([o['mjd'] for o in out]),
		'mag': np.array([o['psf_mag'] for o in out]),
		'magerr': np.array([o['psf_mag_err'] for o in out]),
		'bandid': np.array([o['bandid'] for o in out])}

def wtp_fetch_limits(cur, cand):
	cur.execute('SELECT st.mjdmin AS mjd, sub.bandid AS bandid, sub.limmag AS limmag '
		'FROM subtractions sub INNER JOIN stacks st ON st.stackid = sub.stackid '
		'WHERE sub.field = %s AND numposcand >= 0 AND subid NOT IN '
		'(SELECT subid FROM candidates WHERE q3c_radial_query(ra, dec, %s, %s, %s) '
		'AND ispos = 1);', (cand['field'], cand['ra'], cand['dec'], WTP_CMRADIUS))
	out = cur.fetchall()
	return {'mjd': np.array([o['mjd'] for o in out]),
		'limmag': np.array([o['limmag'] for o in out]),
		'bandid': np.array([o['bandid'] for o in out])}

def wtp_fetch_name(cur, cand):
	# Nearest-first, matching WTP_NAME_JOIN, so figure titles agree with cards
	# when multiple named sources fall within the cone.
	cur.execute('SELECT name FROM sourcenames WHERE q3c_radial_query(ra, dec, %s, %s, %s) '
		'ORDER BY q3c_dist(ra, dec, %s, %s) ASC LIMIT 1;',
		(cand['ra'], cand['dec'], WTP_CMRADIUS, cand['ra'], cand['dec']))
	out = cur.fetchall()
	return out[0]['name'] if out else 'NULL'

def wtp_fetch_cutouts(cur, cand):
	# Show the cutout for the source's brightest-detection candid, as recorded in
	# forced_photometry (constant across its fp rows). Resolve via the source's
	# wtpid; fall back to this candidate's own candid when the source has no
	# forced-photometry entry (unnamed or not yet loaded).
	candid = cand['candid']
	cur.execute('SELECT wtpid FROM sourcenames WHERE q3c_radial_query(ra, dec, %s, %s, %s) '
		'ORDER BY q3c_dist(ra, dec, %s, %s) ASC LIMIT 1;',
		(cand['ra'], cand['dec'], WTP_CMRADIUS, cand['ra'], cand['dec']))
	r = cur.fetchone()
	if r:
		cur.execute('SELECT candid FROM forced_photometry WHERE sourceid = %s LIMIT 1;',
			(r['wtpid'],))
		fr = cur.fetchone()
		if fr and fr['candid'] is not None:
			candid = fr['candid']
	cur.execute('SELECT sci_image, ref_image, diff_image FROM cutouts WHERE candid = %s;',
		(candid,))
	c = cur.fetchone()
	if c is None:
		return None
	def load(blob):
		return np.flipud(fits.open(io.BytesIO(gzip.open(io.BytesIO(blob), 'rb').read()))[0].data)
	return {'sci': load(c['sci_image']), 'ref': load(c['ref_image']),
		'diff': load(c['diff_image']), 'candid': candid}

# Forced-photometry light curve for the source at this candidate's position.
# forced_photometry is keyed by sourceid (indexed) and is NOT spatially indexed,
# so resolve the source's wtpid through sourcenames first, then pull by sourceid.
# Detection := flux SNR >= FP_SNR_THRESH (equivalently forcediffmagpsf <
# diffmaglim, since diffmaglim is the 5-sigma limit); all other epochs are drawn
# as 5-sigma upper limits at diffmaglim. Only named sources were loaded, so an
# unnamed candidate yields None and the viewer falls back to candidate-based.
FP_SNR_THRESH = 5.0

def _farr(rows, key):
	return np.array([np.nan if r[key] is None else float(r[key]) for r in rows],
		dtype=float)

def wtp_fetch_forced(cur, cand):
	cur.execute('SELECT wtpid FROM sourcenames WHERE q3c_radial_query(ra, dec, %s, %s, %s) '
		'ORDER BY q3c_dist(ra, dec, %s, %s) ASC LIMIT 1;',
		(cand['ra'], cand['dec'], WTP_CMRADIUS, cand['ra'], cand['dec']))
	r = cur.fetchone()
	if not r:
		return None
	cur.execute('SELECT mjd, bandid, forcediffimflux, forcediffimfluxunc, '
		'forcediffmagpsf, forcediffsigmapsf, diffmaglim, '
		'forcestackimflux, forcestackimfluxunc, forcestackmagpsf, forcestacksigmapsf '
		'FROM forced_photometry '
		'WHERE sourceid = %s ORDER BY mjd ASC;', (r['wtpid'],))
	out = cur.fetchall()
	if not out:
		return None
	flux = _farr(out, 'forcediffimflux'); func = _farr(out, 'forcediffimfluxunc')
	mag = _farr(out, 'forcediffmagpsf')
	with np.errstate(invalid='ignore', divide='ignore'):
		snr = np.where(func > 0, flux / func, np.nan)
	isdet = np.isfinite(snr) & (snr >= FP_SNR_THRESH) & np.isfinite(mag) & (mag > 0)
	sflux = _farr(out, 'forcestackimflux'); sfunc = _farr(out, 'forcestackimfluxunc')
	smag = _farr(out, 'forcestackmagpsf')
	with np.errstate(invalid='ignore', divide='ignore'):
		ssnr = np.where(sfunc > 0, sflux / sfunc, np.nan)
	sisdet = np.isfinite(ssnr) & (ssnr >= FP_SNR_THRESH) & np.isfinite(smag) & (smag > 0)
	return {'mjd': _farr(out, 'mjd'),
		'bandid': np.array([o['bandid'] for o in out]),
		'mag': mag, 'magerr': _farr(out, 'forcediffsigmapsf'),
		'limmag': _farr(out, 'diffmaglim'), 'isdet': isdet,
		'smag': smag, 'smagerr': _farr(out, 'forcestacksigmapsf'), 'sisdet': sisdet}

# Batched name lookup: nearest sourcenames entry within the match radius,
# attached per candidate by run_candidate_query (column alias names_raw).
WTP_NAME_JOIN = ('LEFT JOIN LATERAL (SELECT name, wtpid FROM sourcenames s '
	'WHERE q3c_join(cand.ra, cand.dec, s.ra, s.dec, %.8f) '
	'ORDER BY q3c_dist(cand.ra, cand.dec, s.ra, s.dec) ASC LIMIT 1) sn ON true'
	% WTP_CMRADIUS)

# --- WTP predicate builders (parameterized; values never interpolated) ------
def _wtp_f(form, k): return float(form[k])
def _wtp_i(form, k): return int(form[k])

def _wtp_base(form):
	# Column refs are qualified with cand. -- xmatch inner-joins source_fp_stats,
	# which has its own columns of these same names, so unqualified refs are
	# ambiguous there even though every other WTP tab only ever sees candidates/fields.
	sql = ('abs(f.gallat) >= %s AND abs(f.gallat) < %s '
		'AND cand.epochid = %s AND cand.rbscore >= %s AND cand.rbscore <= %s '
		'AND cand.nmatches >= %s AND cand.scorr_peak >= %s AND cand.ispos = 1 '
		'AND cand.mjd - cand.firstdet > %s AND cand.mjd - cand.firstdet < %s AND cand.distnearbrstar > %s '
		'AND (cand.wdist1 > %s OR cand.w1mag1 > %s) AND (cand.wdist2 > %s OR cand.w1mag2 > %s) '
		'AND (cand.wdist3 > %s OR cand.w1mag3 > %s)')
	brd, brm = _wtp_f(form, 'brdistlim'), _wtp_f(form, 'brmaglim')
	params = [_wtp_f(form, 'gallimlow'), _wtp_f(form, 'gallimhigh'), _wtp_i(form, 'scanep'),
		_wtp_f(form, 'rbscorelow'), _wtp_f(form, 'rbscorehigh'), _wtp_i(form, 'nmatches'),
		_wtp_f(form, 'scorrpeak'), _wtp_f(form, 'agelowlim'), _wtp_f(form, 'agehighlim'),
		brd, brd, brm, brd, brm, brd, brm]
	return [sql], params

def wtp_hostless(form):
	sql, p = _wtp_base(form)
	hl = _wtp_f(form, 'hlwdist')
	sql.append('wdist1 > %s AND wdist2 > %s AND wdist3 > %s'); p += [hl, hl, hl]
	return ' AND '.join(sql), p

def wtp_hosted(form):
	sql, p = _wtp_base(form)
	hwd, hwa = _wtp_f(form, 'hwdist'), _wtp_f(form, 'hwamp')
	sql.append('wdist1 < %s'); p.append(hwd)
	sql.append('((bandid = 1 AND (w1mag1 - psf_mag > %s) '
		'AND ((w1mag2 - psf_mag > %s) OR (wdist2 > %s)) '
		'AND ((w1mag3 - psf_mag > %s) OR (wdist3 > %s))) '
		'OR (bandid = 2 AND (w2mag1 - psf_mag > %s) '
		'AND ((w2mag2 - psf_mag > %s) OR (wdist2 > %s)) '
		'AND ((w2mag3 - psf_mag > %s) OR (wdist3 > %s))))')
	p += [hwa, hwa, hwd, hwa, hwd, hwa, hwa, hwd, hwa, hwd]
	return ' AND '.join(sql), p

# Nearest NED match attached per candidate; used by the clu/nuclear scans both
# to filter (distance_arcsec < cludist) and to display the host galaxy.
NED_JOIN = ('LEFT JOIN LATERAL (SELECT distance_arcsec, ned_objname, ned_distmpc, '
	'ned_z FROM cand_ned_crossmatch x WHERE x.candid = cand.candid '
	'ORDER BY x.distance_arcsec ASC LIMIT 1) ned ON true')
NED_EXTRA = [('cludists', 'ned.distance_arcsec'),
	('ned_objnames', 'ned.ned_objname'), ('ned_distmpcs', 'ned.ned_distmpc'),
	('ned_zs', 'ned.ned_z')]

def wtp_clu(form):
	sql, p = _wtp_base(form)
	sql.append('ned.distance_arcsec >= 0 AND ned.distance_arcsec < %s')
	p.append(_wtp_f(form, 'cludist'))
	return ' AND '.join(sql), p

# Source-level NED crossmatch (source_ned_crossmatch), keyed by sourcenames.wtpid
# and built with a per-galaxy match radius (10 kpc for the Magellanic Clouds,
# 20 kpc for M31 and distant galaxies, 1 deg cap for Local Group dwarfs).
# Bridged to candidates through sourcenames by position. Used by the nuclear,
# M31, LMC and SMC scans. (extra cols are selected but not surfaced by the
# serializer; they exist for the WHERE filters / future display.)
SNX_EXTRA = [('cludists', 'snx.distance_arcsec'),
	('ned_objnames', 'snx.ned_objname'), ('ned_distmpcs', 'snx.ned_distmpc'),
	('ned_zs', 'snx.ned_z'), ('proj_sep_kpcs', 'snx.proj_sep_kpc')]

def _wtp_snx_join(objname=None):
	"""LATERAL attaching the candidate's source-level NED match from
	source_ned_crossmatch (via sourcenames.wtpid). With objname set, restrict to
	that NED galaxy; otherwise attach the nearest matched galaxy."""
	name_filter = 'AND scn.ned_objname = %s ' if objname else ''
	join = ('LEFT JOIN LATERAL (SELECT scn.ned_objname, scn.ned_distmpc, scn.ned_z, '
		'scn.distance_arcsec, scn.proj_sep_kpc FROM source_ned_crossmatch scn '
		'WHERE scn.wtpid = sn.wtpid ' + name_filter +
		'ORDER BY scn.distance_arcsec ASC LIMIT 1) snx ON true')
	return join, ([objname] if objname else [])

def wtp_nuclear(form):
	# Transient within cludist arcsec of a NED galaxy centre (nuclear flares).
	sql, p = _wtp_base(form)
	sql.append('snx.distance_arcsec >= 0 AND snx.distance_arcsec < %s')
	p.append(_wtp_f(form, 'cludist'))
	return ' AND '.join(sql), p

# Bounding region (centre + radius) of a named galaxy's matched sources, from
# source_ned_crossmatch; cached per process. Used as a candidate-side q3c cone
# prefilter so the membership LATERAL runs only over the galaxy's neighbourhood
# instead of the whole epoch (the cause of the original slowness).
_GALAXY_REGION = {}

def _galaxy_region(objname):
	"""(ra, dec, radius_deg) enclosing every member candidate, or None. radius =
	max member-to-centre separation + cmradius."""
	if objname in _GALAXY_REGION:
		return _GALAXY_REGION[objname]
	conn, cur = _wtp_open()
	try:
		cur.execute('SELECT ned_ra, ned_dec, max(distance_arcsec) AS maxsep '
			'FROM source_ned_crossmatch WHERE ned_objname = %s '
			'GROUP BY ned_ra, ned_dec ORDER BY maxsep DESC LIMIT 1;', (objname,))
		r = cur.fetchone()
	finally:
		_wtp_close(conn, cur)
	region = None if r is None else (float(r['ned_ra']), float(r['ned_dec']),
		float(r['maxsep']) / 3600.0 + WTP_CMRADIUS)
	_GALAXY_REGION[objname] = region
	return region

def wtp_galaxy_member(form, objname):
	# Candidate-side q3c cone around the galaxy prefilters via the candidates
	# spatial index; the LATERAL then confirms exact membership. Plus the
	# hostless WISE-distance cut.
	sql, p = _wtp_base(form)
	region = _galaxy_region(objname)
	if region is not None:
		ra0, dec0, rad = region
		sql.append('q3c_radial_query(ra, dec, %s, %s, %s)')
		p += [ra0, dec0, rad]
	hl = _wtp_f(form, 'hlwdist')
	sql.append('snx.ned_objname IS NOT NULL '
		'AND wdist1 > %s AND wdist2 > %s AND wdist3 > %s')
	p += [hl, hl, hl]
	return ' AND '.join(sql), p

def _wtp_radial(form, ra, dec, dist_key):
	sql, p = _wtp_base(form)
	hl = _wtp_f(form, 'hlwdist')
	sql.append('q3c_radial_query(ra, dec, %s, %s, %s) '
		'AND wdist1 > %s AND wdist2 > %s AND wdist3 > %s')
	p += [ra, dec, _wtp_f(form, dist_key), hl, hl, hl]
	return ' AND '.join(sql), p

def wtp_lmc(form):
	# LMC membership (set in the join) + per-WISE-neighbour hostless/amplitude
	# logic, with a candidate-side cone prefilter (see wtp_galaxy_member).
	sql, p = _wtp_base(form)
	region = _galaxy_region('Large Magellanic Cloud')
	if region is not None:
		ra0, dec0, rad = region
		sql.append('q3c_radial_query(ra, dec, %s, %s, %s)')
		p += [ra0, dec0, rad]
	hl, hwd, hwa = _wtp_f(form, 'hlwdist'), _wtp_f(form, 'hwdist'), _wtp_f(form, 'hwamp')
	sql.append('snx.ned_objname IS NOT NULL')
	for n in (1, 2, 3):
		sql.append('(wdist%d > %%s OR (wdist%d < %%s AND '
			'((bandid = 1 AND (w1mag%d - psf_mag > %%s)) '
			'OR (bandid = 2 AND (w2mag%d - psf_mag > %%s)))))' % (n, n, n, n))
		p += [hl, hwd, hwa, hwa]
	return ' AND '.join(sql), p

def _wtp_yso_join(form):
	"""Per-candidate cone stats in SQL, replacing the old Python batch loop:
	W1 bright-phase duration and per-epoch-median W1-W2 colour (cmradius cone).
	Returns (join_sql, join_params); params bind before the WHERE params."""
	durmag = _wtp_f(form, 'durmag')
	join = ('LEFT JOIN LATERAL (SELECT '
		'max(c2.mjd) FILTER (WHERE c2.bandid = 1 AND c2.psf_mag < %%s) - '
		'min(c2.mjd) FILTER (WHERE c2.bandid = 1 AND c2.psf_mag < %%s) AS brightdur '
		'FROM candidates c2 WHERE q3c_join(cand.ra, cand.dec, c2.ra, c2.dec, %.8f) '
		'AND c2.ispos = 1) dur ON true '
		'LEFT JOIN LATERAL (SELECT percentile_cont(0.5) WITHIN GROUP '
		'(ORDER BY ee.w1med - ee.w2med) AS medcolor FROM '
		'(SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY c3.psf_mag) '
		'FILTER (WHERE c3.bandid = 1) AS w1med, '
		'percentile_cont(0.5) WITHIN GROUP (ORDER BY c3.psf_mag) '
		'FILTER (WHERE c3.bandid = 2) AS w2med '
		'FROM candidates c3 WHERE q3c_join(cand.ra, cand.dec, c3.ra, c3.dec, %.8f) '
		'AND c3.ispos = 1 GROUP BY c3.epochid) ee '
		'WHERE ee.w1med IS NOT NULL AND ee.w2med IS NOT NULL) col ON true'
		% (WTP_CMRADIUS, WTP_CMRADIUS))
	return join, [durmag, durmag]

def wtp_yso(form):
	minduration = _wtp_f(form, 'minduration') * 365
	brd, brm = _wtp_f(form, 'brdistlim'), _wtp_f(form, 'brmaglim')
	sql = ('abs(f.gallat) >= %s AND abs(f.gallat) < %s AND epochid = %s '
		'AND rbscore >= %s AND rbscore <= %s AND dec > %s AND dec < %s '
		'AND ra > %s AND ra < %s AND scorr_peak >= %s AND psf_mag >= %s AND psf_mag < %s '
		'AND ispos = 1 AND mjd - firstdet > %s AND distnearbrstar > %s '
		'AND (wdist1 > %s OR w1mag1 > %s) AND (wdist2 > %s OR w1mag2 > %s) '
		'AND (wdist3 > %s OR w1mag3 > %s) '
		'AND dur.brightdur > %s AND col.medcolor >= %s AND col.medcolor <= %s')
	params = [_wtp_f(form, 'gallimlow'), _wtp_f(form, 'gallimhigh'),
		_wtp_i(form, 'scanep'), _wtp_f(form, 'rbscorelow'),
		_wtp_f(form, 'rbscorehigh'), _wtp_f(form, 'declimlow'),
		_wtp_f(form, 'declimhigh'), _wtp_f(form, 'ralimlow'),
		_wtp_f(form, 'ralimhigh'), _wtp_f(form, 'scorrpeak'),
		_wtp_f(form, 'curmaglow'), _wtp_f(form, 'curmaghigh'), minduration, brd, brd, brm, brd, brm, brd, brm,
		minduration, _wtp_f(form, 'colorlimlow'), _wtp_f(form, 'colorlimhigh')]
	return sql, params

# Form field specs consumed by the single scan.html template. 'num' -> one input,
# 'range' -> lo/hi pair, 'brstar' -> the bright-star rejection row.
_F_BASE = [
	{'kind': 'num', 'label': 'Epoch', 'name': 'scanep'},
	{'kind': 'range', 'label': 'RB score', 'lo': 'rbscorelow', 'hi': 'rbscorehigh'},
	{'kind': 'num', 'label': 'NMatches \u2265', 'name': 'nmatches'},
	{'kind': 'num', 'label': 'Scorr peak \u2265', 'name': 'scorrpeak'},
	{'kind': 'range', 'label': '|Gal. lat|', 'lo': 'gallimlow', 'hi': 'gallimhigh', 'suffix': '\u00b0'},
	{'kind': 'range', 'label': 'Age', 'lo': 'agelowlim', 'hi': 'agehighlim', 'suffix': 'days'},
]
_F_BR = {'kind': 'brstar'}

_F_YSO = [
	{'kind': 'num', 'label': 'Epoch', 'name': 'scanep'},
	{'kind': 'range', 'label': 'RB score', 'lo': 'rbscorelow', 'hi': 'rbscorehigh'},
	{'kind': 'range', 'label': '|Gal. lat|', 'lo': 'gallimlow', 'hi': 'gallimhigh', 'suffix': '\u00b0'},
	{'kind': 'range', 'label': 'Dec', 'lo': 'declimlow', 'hi': 'declimhigh', 'suffix': '\u00b0'},
	{'kind': 'range', 'label': 'RA', 'lo': 'ralimlow', 'hi': 'ralimhigh', 'suffix': '\u00b0'},
	{'kind': 'num', 'label': 'Scorr peak \u2265', 'name': 'scorrpeak'},
	{'kind': 'range', 'label': 'Epoch brightness', 'lo': 'curmaglow', 'hi': 'curmaghigh', 'suffix': 'mag'},
	{'kind': 'pair', 'label': 'Bright \u2264', 'name': 'durmag', 'suffix': 'mag', 'mid': '\u2026for over', 'name2': 'minduration', 'suffix2': 'yr (W1)'},
	{'kind': 'range', 'label': 'W1\u2212W2 colour', 'lo': 'colorlimlow', 'hi': 'colorlimhigh'},
	_F_BR,
]

WTP_FIELD_MAP = [
	('candids', 'candid'), ('mjds', 'mjd'), ('firstdets', 'firstdet'),
	('nmatches', 'nmatches'), ('rbscores', 'rbscore'), ('psf_mags', 'psf_mag'),
	('bandids', 'bandid'), ('distnearbrstars', 'distnearbrstar'),
	('magnearbrstars', 'magnearbrstar'),
	('wdist1', 'wdist1'), ('w1mag1', 'w1mag1'), ('wdist2', 'wdist2'), ('w1mag2', 'w1mag2'),
	('wdist3', 'wdist3'), ('w1mag3', 'w1mag3'),
	('w2mag1', 'w2mag1'), ('w2mag2', 'w2mag2'), ('w2mag3', 'w2mag3'),
	('scorrpeaks', 'scorr_peak'), ('ras', 'ra'), ('decs', 'dec'),
]

WTP = Survey(
	key='wtp', label='WTP (unWISE W1/W2)',
	open_cursor=_wtp_open, close_cursor=_wtp_close,
	base_from='candidates cand INNER JOIN fields f ON f.field = cand.field',
	alias='cand', id_col='candid',
	bands=[Band(1, 'W1', 'k'), Band(2, 'W2', 'r')],
	field_map=WTP_FIELD_MAP,
	allowed_sort={'candid', 'mjd', 'firstdet', 'nmatches', 'rbscore', 'psf_mag',
		'scorr_peak', 'distnearbrstar', 'wdist1', 'wdist2', 'wdist3'},
	default_sort=('scorr_peak', 'DESC'),
	scans={
		'hostless': {'template': 'scan.html', 'builder': wtp_hostless,
			'title': 'Hostless transients',
			'fields': _F_BASE + [
				{'kind': 'num', 'label': 'Min. WISE-source dist', 'name': 'hlwdist', 'suffix': '\u2033'},
				_F_BR,
			]},
		'hosted': {'template': 'scan.html', 'builder': wtp_hosted,
			'title': 'Hosted transients',
			'fields': _F_BASE + [
				{'kind': 'num', 'label': 'Max. WISE-source dist', 'name': 'hwdist', 'suffix': '\u2033'},
				{'kind': 'num', 'label': 'Amplitude vs AllWISE \u2265', 'name': 'hwamp', 'suffix': 'mag'},
				_F_BR,
			]},
		'clu': {'template': 'scan.html', 'builder': wtp_nuclear,
			'join': lambda f: _wtp_snx_join(), 'extra': SNX_EXTRA,
			'title': 'Nearby-galaxy (CLU) transients',
			'fields': _F_BASE + [
				{'kind': 'num', 'label': 'NED match dist <', 'name': 'cludist', 'suffix': '\u2033'},
				_F_BR,
			]},
		'nuclear': {'template': 'scan.html', 'builder': wtp_nuclear,
			'join': lambda f: _wtp_snx_join(), 'extra': SNX_EXTRA,
			'title': 'IR nuclear flares',
			'fields': _F_BASE + [
				{'kind': 'num', 'label': 'NED match dist <', 'name': 'cludist', 'suffix': '\u2033'},
				_F_BR,
			]},
		'smc': {'template': 'scan.html', 'builder': lambda f: wtp_galaxy_member(f, 'Small Magellanic Cloud'),
			'join': lambda f: _wtp_snx_join('Small Magellanic Cloud'), 'extra': SNX_EXTRA,
			'title': 'SMC transients',
			'fields': _F_BASE + [
				{'kind': 'num', 'label': 'Hostless WISE dist \u2265', 'name': 'hlwdist', 'suffix': '\u2033'},
				_F_BR,
			]},
		'm31': {'template': 'scan.html', 'builder': lambda f: wtp_galaxy_member(f, 'MESSIER 031'),
			'join': lambda f: _wtp_snx_join('MESSIER 031'), 'extra': SNX_EXTRA,
			'title': 'M31 transients',
			'fields': _F_BASE + [
				{'kind': 'num', 'label': 'Hostless WISE dist \u2265', 'name': 'hlwdist', 'suffix': '\u2033'},
				_F_BR,
			]},
		'lmc': {'template': 'scan.html', 'builder': wtp_lmc,
			'join': lambda f: _wtp_snx_join('Large Magellanic Cloud'), 'extra': SNX_EXTRA,
			'title': 'LMC transients',
			'fields': _F_BASE + [
				{'kind': 'num', 'label': 'Hostless WISE dist \u2265', 'name': 'hlwdist', 'suffix': '\u2033'},
				{'kind': 'num', 'label': '\u2026or amplitude \u2265', 'name': 'hwamp', 'suffix': 'mag'},
				{'kind': 'num', 'label': '\u2026within', 'name': 'hwdist', 'suffix': '\u2033'},
				_F_BR,
			]},
		'yso': {'template': 'scan.html', 'builder': wtp_yso,
			'join': _wtp_yso_join,
			'title': 'Young-star outbursts',
			'fields': _F_YSO},
	},
	fetch_lightcurve=wtp_fetch_lightcurve, fetch_limits=wtp_fetch_limits,
	fetch_name=wtp_fetch_name, fetch_cutouts=wtp_fetch_cutouts,
	fetch_forced=wtp_fetch_forced,
	fetch_saved_names=wtp_fritz_saved_names, cmradius=WTP_CMRADIUS,
	name_join=WTP_NAME_JOIN,
)

# ============================================================================
# ADAPTER 2: PRIME (NIR, Z/Y/J/H)  --  TEMPLATE. Replace the *** markers with
# your real PRIME schema. Structure is complete; only column/table names and
# the predicate logic need to match the PRIME database.
# ============================================================================
# PRIME uses the pooled adapter (see make_pool_adapter above). Connections are
# created lazily on first scan; set PRIME_DSN and optionally PRIME_POOL_MAX.
_prime_open, _prime_close = make_pool_adapter('prime', 'PRIME_DSN',
	maxconn=int(os.environ.get('PRIME_POOL_MAX', '8')))

PRIME_CMRADIUS = 2.0 / 3600		# *** PRIME astrometric match radius

def _prime_f(form, k): return float(form[k])
def _prime_i(form, k): return int(form[k])

def prime_transient(form):
	# *** EXAMPLE generic transient cut -- replace columns with PRIME's.
	# No WISE-neighbor columns here; PRIME has its own host/quality columns.
	sql = ('rbscore >= %s AND rbscore <= %s AND scorr_peak >= %s AND ispos = 1 '
		'AND mjd - firstdet > %s AND mjd - firstdet < %s AND distnearbrstar > %s')
	params = [_prime_f(form, 'rbscorelow'), _prime_f(form, 'rbscorehigh'),
		_prime_f(form, 'scorrpeak'), _prime_f(form, 'agelowlim'),
		_prime_f(form, 'agehighlim'), _prime_f(form, 'brdistlim')]
	return sql, params

def prime_spatial(form):
	# Cone search around an arbitrary center -- works for the bulge or any region
	sql = 'q3c_radial_query(ra, dec, %s, %s, %s)'
	params = [_prime_f(form, 'rasearch'), _prime_f(form, 'decsearch'),
		_prime_f(form, 'cmradius') / 3600]
	return sql, params

PRIME_FIELD_MAP = [
	# *** template keys your PRIME templates consume; map to real PRIME columns
	('candids', 'candid'), ('mjds', 'mjd'), ('firstdets', 'firstdet'),
	('rbscores', 'rbscore'), ('psf_mags', 'psf_mag'), ('bandids', 'bandid'),
	('scorrpeaks', 'scorr_peak'), ('distnearbrstars', 'distnearbrstar'),
	('ras', 'ra'), ('decs', 'dec'),
]

PRIME = Survey(
	key='prime', label='PRIME (NIR Z/Y/J/H)',
	open_cursor=_prime_open, close_cursor=_prime_close,
	base_from='candidates cand',				# *** add JOINs if PRIME needs them
	alias='cand', id_col='candid',
	bands=[Band(1, 'Z', 'purple'), Band(2, 'Y', 'blue'),	# *** real PRIME band ids
		Band(3, 'J', 'green'), Band(4, 'H', 'red')],
	field_map=PRIME_FIELD_MAP,
	allowed_sort={'candid', 'mjd', 'firstdet', 'rbscore', 'psf_mag',
		'scorr_peak', 'distnearbrstar'},
	default_sort=('scorr_peak', 'DESC'),
	scans={
		'transient': {'template': 'prime_transient_scan.html', 'builder': prime_transient},
		'spatial': {'template': 'prime_spatial_scan.html', 'builder': prime_spatial},
	},
	fetch_lightcurve=None, fetch_limits=None,
	fetch_name=None, fetch_cutouts=None,
	cmradius=PRIME_CMRADIUS,
)

# ----------------------------------------------------------------------------
SURVEYS.update({WTP.key: WTP, PRIME.key: PRIME})


# >>> PRIME hostless + large-amplitude scans
# ---------------------------------------------------------------------------
# Appended after the PRIME Survey is constructed; mutates PRIME in place.
# band lives in filter2 (filter1 is always 'Open'); valid filter2 = Y/J/H.
# SCANS are H-band only; LIGHT CURVES show all bands. The form takes a UTC date
# range (datemin/datemax) converted to MJD for the query. hostless = no VVV
# source within hlwdist; largeamp = VVV host within hwdist and the H-band
# transient brighter than the VVV H host by >= hwamp mag. Results render via
# prime_scan.html / prime_candidate_cards. Overrides the bandid-based stub.
# ---------------------------------------------------------------------------

PRIME_FILTER_BAND = {'Y': 1, 'J': 2, 'H': 3}
PRIME.bands = [Band(1, 'Y', 'darkorange'), Band(2, 'J', 'm'), Band(3, 'H', 'k')]

PRIME.cmradius = PRIME_CMRADIUS

def prime_fetch_lightcurve(cur, cand):
    # Forced-only: scans surface only sources that have forced photometry
    # (see PRIME_NAME_JOIN's EXISTS), and the plot shows the forced series
    # exclusively. Return no candidate points -- there is NO candidate fallback.
    empty = np.array([])
    return {'mjd': empty, 'mag': empty, 'magerr': empty, 'bandid': empty}
PRIME.fetch_lightcurve = prime_fetch_lightcurve

def prime_fetch_name(cur, cand):
    # Real source name from the sources catalog (nearest by position).
    cur.execute('SELECT name FROM sources '
        'WHERE q3c_radial_query(ra, dec, %s, %s, %s) '
        'ORDER BY q3c_dist(ra, dec, %s, %s) ASC LIMIT 1;',
        (cand['ra'], cand['dec'], PRIME_CMRADIUS, cand['ra'], cand['dec']))
    r = cur.fetchone()
    return r['name'] if r else ('PRIME%d' % cand['candid'])
PRIME.fetch_name = prime_fetch_name

def prime_fetch_cutouts(cur, cand):
    # cutouts is keyed by sourcename (one stamp per source), NOT by the per-epoch
    # candid, so resolve this candidate to its nearest source by position (sources
    # has the q3c index), then fetch by sourcename. Decode tolerates gzipped or
    # raw FITS bytes.
    cur.execute('SELECT name FROM sources '
        'WHERE q3c_radial_query(ra, dec, %s, %s, %s) '
        'ORDER BY q3c_dist(ra, dec, %s, %s) ASC LIMIT 1;',
        (cand['ra'], cand['dec'], PRIME_CMRADIUS, cand['ra'], cand['dec']))
    r = cur.fetchone()
    if not r:
        return None
    cur.execute('SELECT candid, mjd, sci_image, ref_image, diff_image FROM cutouts '
        'WHERE sourcename = %s;', (r['name'],))
    c = cur.fetchone()
    if c is None:
        return None
    def load(blob):
        b = bytes(blob)
        if b[:2] == b'\x1f\x8b':            # gzip magic -> decompress first
            b = gzip.open(io.BytesIO(b), 'rb').read()
        return np.flipud(fits.open(io.BytesIO(b))[0].data)
    return {'sci': load(c['sci_image']), 'ref': load(c['ref_image']),
        'diff': load(c['diff_image']), 'candid': c['candid'], 'mjd': c['mjd']}
PRIME.fetch_cutouts = prime_fetch_cutouts

PRIME_FP_SNR = 5.0       # forced-phot detection threshold: flux / forcediffimfluxstaterr
PRIME_FP_MAGFLOOR = 20.0 # drop forced points fainter than this (tiny-error artifacts)

def prime_fetch_forced(cur, cand):
    # Forced-photometry light curve for this candidate's source (built nightly
    # by make_prime_sources_fp.py), keyed by primeid. Resolve the source by
    # position, then pull every epoch. Detection := flux SNR >= PRIME_FP_SNR
    # using the formal flux error forcediffimfluxstaterr (the same error
    # forcediffsigmapsf is derived from); non-detections are drawn as upper
    # limits at diffmaglim. Reuses _farr from the WTP section (None -> NaN).
    cur.execute('SELECT primeid FROM sources '
        'WHERE q3c_radial_query(ra, dec, %s, %s, %s) '
        'ORDER BY q3c_dist(ra, dec, %s, %s) ASC LIMIT 1;',
        (cand['ra'], cand['dec'], PRIME_CMRADIUS, cand['ra'], cand['dec']))
    r = cur.fetchone()
    if not r:
        return None
    cur.execute('SELECT mjd, filter2, forcediffimflux, forcediffimfluxstaterr, '
        'forcediffmagpsf, forcediffsigmapsf, diffmaglim '
        'FROM forced_photometry WHERE primeid = %s ORDER BY mjd ASC;', (r['primeid'],))
    out = cur.fetchall()
    if not out:
        return None
    flux = _farr(out, 'forcediffimflux'); ferr = _farr(out, 'forcediffimfluxstaterr')
    mag = _farr(out, 'forcediffmagpsf'); lim = _farr(out, 'diffmaglim')
    mjd = _farr(out, 'mjd'); magerr = _farr(out, 'forcediffsigmapsf')
    bid = np.array([PRIME_FILTER_BAND.get(o['filter2'], 0) for o in out])
    with np.errstate(invalid='ignore', divide='ignore'):
        snr = np.where(ferr > 0, flux / ferr, np.nan)
    isdet = np.isfinite(snr) & (snr >= PRIME_FP_SNR) & np.isfinite(mag) & (mag > 0)
    # Drop any epoch whose PLOTTED magnitude is fainter than the floor -- the
    # measurement for a detection, the limit (limmag) for a non-detection. This
    # kills both bogus faint detections AND unrealistically deep limit arrows
    # (the >20 mag J/Y upper limits from tiny forced errors).
    shown = np.where(isdet, mag, lim)
    keep = ~(np.isfinite(shown) & (shown > PRIME_FP_MAGFLOOR))
    return {'mjd': mjd[keep], 'bandid': bid[keep], 'mag': mag[keep],
        'magerr': magerr[keep], 'limmag': lim[keep], 'isdet': isdet[keep]}
PRIME.fetch_forced = prime_fetch_forced

# Batched name lookup from the sources catalog (nearest within cmradius),
# attached per candidate as names_raw -- one indexed join instead of a q3c
# lookup per row. Also enables the engine's DISTINCT ON dedup, so each source
# shows once (its extreme-in-sort epoch) rather than one card per epoch.
# INNER lateral join: only candidates within cmradius of a NAMED source
# survive -- unnamed candidates are excluded from PRIME scans entirely. A
# survivor is by construction <cmradius from its source, so the cutout/name
# lookups always resolve.
PRIME_NAME_JOIN = ('JOIN LATERAL (SELECT s.name FROM sources s '
    'WHERE q3c_join(cand.ra, cand.dec, s.ra, s.dec, %.8f) '
    'AND EXISTS (SELECT 1 FROM forced_photometry fp WHERE fp.primeid = s.primeid) '
    'ORDER BY q3c_dist(cand.ra, cand.dec, s.ra, s.dec) ASC LIMIT 1) sn ON true'
    % PRIME_CMRADIUS)
PRIME.name_join = PRIME_NAME_JOIN

PRIME.field_map = [
    ('candids', 'candid'), ('mjds', 'mjd'), ('firstdets', 'firstdet'),
    ('rbscores', 'rbscore'), ('psf_mags', 'psf_mag'), ('filters', 'filter2'),
    ('fields', 'field'), ('fpaposs', 'fpapos'),
    ('scorrpeaks', 'scorr_peak'),
    ('distnearbrstars', 'distnearbrstar'), ('magnearbrstars', 'magnearbrstar'),
    ('vvvdist1', 'vvvdist1'), ('vvvmagh1', 'vvvmagh1'),
    ('tmdist1', 'tmdist1'), ('tmmagh1', 'tmmagh1'),
    ('numnegpixs', 'numnegpix'),
    ('numbadweightscis', 'numbadweightsci'),
    ('numbadweightrefs', 'numbadweightref'),
    ('psf_chi2s', 'psf_chi2'), ('fwhms', 'fwhm'), ('sumrats', 'sumrat'),
    ('nmatches', 'nmatches'),
    ('neglobe_clusts', 'neglobe_clust'), ('neglobe_dists', 'neglobe_dist'),
    ('ras', 'ra'), ('decs', 'dec'),
]

# UTC date <-> MJD
PRIME_DATE_LO = '2023-01-01'
PRIME_DATE_HI = Time.now().iso[:10]

def _prime_mjd_bound(form, key, default, end=False):
    """UTC 'YYYY-MM-DD' (date picker) or bare MJD -> MJD; blank -> default.
    end=True -> end of that UTC day (inclusive upper bound)."""
    s = str(form.get(key, '') or default).strip()
    if '-' in s:
        mjd = Time(s, scale='utc').mjd
        return mjd + 1.0 if end else mjd
    return float(s)


# >>> PRIME flux panel + sex coords
# Adds (i) a flux subpanel under the mag panel in PRIME light curves and
# (ii) a per-band staterr outlier filter, by wrapping candidate_png_by_id for
# PRIME only. WTP path is unchanged. Sex coords on cards are added by wrapping
# build_scan_context: it post-annotates canddicts with a 'coords_sex' list.
PRIME_FLUX_ERR_MAX_FACTOR = 3.0   # drop epochs with staterr > factor x band median

def _prime_band_outlier_mask(bid, ferr):
    """True for "keep". Per band, drop epochs whose forcediffimfluxstaterr exceeds
    PRIME_FLUX_ERR_MAX_FACTOR x that band's median (in-band scale, not global)."""
    keep = np.isfinite(ferr) & (ferr > 0)
    for b in np.unique(bid[keep]):
        sel = (bid == b) & keep
        med = float(np.median(ferr[sel]))
        if not np.isfinite(med) or med <= 0:
            continue
        keep &= ~((bid == b) & (ferr > PRIME_FLUX_ERR_MAX_FACTOR * med))
    return keep

def _prime_render_png(survey, candid):
    """PRIME light-curve PNG: 3 cutout panels on top, mag panel middle, flux
    panel bottom (both panels share x-axis = MJD). Uses survey.fetch_forced
    directly for the flux series so it sees raw flux/staterr in addition to the
    mag/limmag the mag panel needs."""
    import io as _io
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from astropy.time import Time as _Time
    from astropy.stats import sigma_clipped_stats as _scs

    conn, cur = survey.open_cursor()
    try:
        cur.execute('SELECT %s.* FROM %s WHERE %s.%s = %%s LIMIT 1'
            % (survey.alias, survey.base_from, survey.alias, survey.id_col),
            (candid,))
        cand = cur.fetchone()
        if cand is None:
            raise KeyError(candid)
        name = survey.fetch_name(cur, cand) if survey.fetch_name else 'NULL'
        cut = survey.fetch_cutouts(cur, cand) if survey.fetch_cutouts else None

        # Pull a raw FP dict that ALSO carries flux/staterr (the standard
        # fetch_forced strips them). Inline the query so we don't depend on it.
        cur.execute('SELECT primeid FROM sources '
            'WHERE q3c_radial_query(ra, dec, %s, %s, %s) '
            'ORDER BY q3c_dist(ra, dec, %s, %s) ASC LIMIT 1;',
            (cand['ra'], cand['dec'], PRIME_CMRADIUS,
             cand['ra'], cand['dec']))
        r = cur.fetchone()
        if r:
            cur.execute('SELECT mjd, filter2, forcediffimflux, forcediffimfluxstaterr, '
                'forcediffmagpsf, forcediffsigmapsf, diffmaglim '
                'FROM forced_photometry WHERE primeid = %s ORDER BY mjd ASC;',
                (r['primeid'],))
            out = cur.fetchall()
        else:
            out = []
    finally:
        survey.close_cursor(conn, cur)

    from matplotlib.gridspec import GridSpec
    if cut is not None:
        fig = Figure(figsize=(10, 9))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.1, 1.2, 1.2],
            hspace=0.45, wspace=0.05)
        axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
        ax_mag = fig.add_subplot(gs[1, :])
        ax_flx = fig.add_subplot(gs[2, :], sharex=ax_mag)
        for a, img, ttl in zip(axes, (cut['sci'], cut['ref'], cut['diff']),
                ('Science', 'Reference', 'Difference')):
            _, med, std = _scs(img)
            a.imshow(img, cmap='gray', vmin=med - std, vmax=med + 5 * std,
                aspect='equal')
            a.set_title(ttl, fontsize=14)
            a.set_xticks([]); a.set_yticks([])
    else:
        fig = Figure(figsize=(10, 6))
        gs = GridSpec(2, 1, figure=fig, hspace=0.35)
        ax_mag = fig.add_subplot(gs[0])
        ax_flx = fig.add_subplot(gs[1], sharex=ax_mag)

    if out:
        mjd_a = _farr(out, 'mjd')
        bid_a = np.array([PRIME_FILTER_BAND.get(o['filter2'], 0) for o in out])
        flux = _farr(out, 'forcediffimflux')
        ferr = _farr(out, 'forcediffimfluxstaterr')
        mag = _farr(out, 'forcediffmagpsf')
        magerr = _farr(out, 'forcediffsigmapsf')
        lim = _farr(out, 'diffmaglim')

        # Outlier mask applies to BOTH panels.
        keep = _prime_band_outlier_mask(bid_a, ferr)

        with np.errstate(invalid='ignore', divide='ignore'):
            snr = np.where(ferr > 0, flux / ferr, np.nan)
        isdet = (np.isfinite(snr) & (snr >= PRIME_FP_SNR)
            & np.isfinite(mag) & (mag > 0))

        # Mag panel: same suppression as prime_fetch_forced (floor on plotted value).
        shown = np.where(isdet, mag, lim)
        mag_keep = keep & ~(np.isfinite(shown) & (shown > PRIME_FP_MAGFLOOR))
        for b in survey.bands:
            sel = mag_keep & (bid_a == b.id)
            if not np.any(sel):
                continue
            det = sel & isdet
            lm = sel & ~isdet & np.isfinite(lim)
            if np.any(det):
                ax_mag.errorbar(mjd_a[det], mag[det], yerr=np.abs(magerr[det]),
                    ls='none', marker='o', color=b.color, ms=10, label=b.label)
            if np.any(lm):
                ax_mag.errorbar(mjd_a[lm], lim[lm], yerr=0.2, uplims=True,
                    ls='none', marker='v', color=b.color,
                    markerfacecolor='none', ms=8)

        # Flux panel: ALL kept epochs (signed), errorbars from staterr.
        for b in survey.bands:
            sel = keep & (bid_a == b.id)
            if not np.any(sel):
                continue
            ax_flx.errorbar(mjd_a[sel], flux[sel], yerr=ferr[sel], ls='none',
                marker='o', color=b.color, ms=7, label=b.label)
        ax_flx.axhline(0.0, color='0.5', lw=0.8)

    ax_mag.invert_yaxis()
    ax_mag.set_ylabel('Mag', fontsize=14)
    ax_mag.tick_params(labelsize=11)
    ax_mag.set_title('Candidate %d; %s' % (candid, name), fontsize=14)
    if ax_mag.get_legend_handles_labels()[1]:
        ax_mag.legend(fontsize=10)
    ax_flx.set_ylabel('Flux (diff)', fontsize=14)
    ax_flx.set_xlabel('MJD', fontsize=14)
    ax_flx.tick_params(labelsize=11)

    # Epoch markers (same convention as WTP): blue = candidate, red = cutout.
    for axx in (ax_mag, ax_flx):
        cmjd = cand.get('mjd')
        if cmjd is not None and np.isfinite(float(cmjd)):
            axx.axvline(float(cmjd), color='b', ls='--', lw=1.0, alpha=0.7)
        xmjd = cut.get('mjd') if cut is not None else None
        if xmjd is not None and np.isfinite(float(xmjd)):
            axx.axvline(float(xmjd), color='r', ls='--', lw=1.0, alpha=0.7)

    # Year twin on the mag panel (above), if any data span exists.
    if out:
        ax_yr = ax_mag.twiny()
        ax_yr.set_xlim(_Time(ax_mag.get_xlim(), format='mjd').decimalyear)
        ax_yr.set_xlabel('Year', fontsize=12)
        ax_yr.tick_params(labelsize=10)

    fig.tight_layout()
    FigureCanvasAgg(fig)
    buf = _io.BytesIO()
    fig.savefig(buf, format='png', dpi=80)
    return buf.getvalue()

# Dispatch wrapper -- only PRIME goes through the flux-panel renderer.
_orig_candidate_png_by_id = candidate_png_by_id
def candidate_png_by_id(survey, candid):
    if getattr(survey, 'key', None) == 'prime':
        return _prime_render_png(survey, candid)
    return _orig_candidate_png_by_id(survey, candid)

# Wrap build_scan_context to inject sexagesimal coords for PRIME results.
_orig_build_scan_context = build_scan_context
def build_scan_context(survey, scan_name, form):
    template, ctx = _orig_build_scan_context(survey, scan_name, form)
    try:
        if getattr(survey, 'key', None) != 'prime':
            return template, ctx
        cd = ctx.get('canddicts')
        if not cd or not len(cd.get('ras', [])):
            return template, ctx
        try:
            sk = SkyCoord(ra=cd['ras'], dec=cd['decs'], unit='degree', frame='icrs')
            cd['coords_sex'] = list(sk.to_string('hmsdms', precision=2))
        except Exception as _e:
            import logging
            logging.getLogger(__name__).warning('prime sex coords failed: %s', _e)
            cd['coords_sex'] = [''] * len(cd['ras'])
    except Exception as _e:
        import logging
        logging.getLogger(__name__).exception('build_scan_context wrapper: %s', _e)
    return template, ctx
# <<< PRIME flux panel + sex coords

# >>> PRIME nightly scan (Layer 0 age scope + Layer 1 shape cuts @ T@98)
# Thresholds are read from the frozen cuts JSON ($PRIME_CUTS_JSON, i.e.
# nightly_cuts.json from test_nightly_layers.py; derivation 2026-07-08,
# validated GB74/C1, band H). Code-level values below are FALLBACKS only --
# the JSON is the source of truth (masterKeys lesson: change the file, not
# the code). Every threshold remains a form field, prefilled with the JSON
# value, so it can be loosened interactively; the "SQL query used" panel
# shows the live numbers.
# Layer 2 (source_history join: run_prior veto, z columns, confirmed badge)
# is a later, purely additive stage -- deliberately NOT included here.
import json as _json

_NC_PATH = os.environ.get('PRIME_CUTS_JSON', '')
_NC = None
if _NC_PATH:
    try:
        with open(_NC_PATH) as _fh:
            _NC = _json.load(_fh)
        logging.getLogger(__name__).info('PRIME nightly cuts loaded from %s', _NC_PATH)
    except Exception as _e:
        logging.getLogger(__name__).warning(
            'PRIME_CUTS_JSON unreadable (%s); using code fallbacks', _e)


def _nc(path, default):
    """Dotted lookup into the cuts JSON; falls back when absent/unreadable.
    Warns when the JSON loaded but a path is missing, so key-name drift
    between nightly_cuts.json and the code is visible at startup."""
    d = _NC if _NC is not None else {}
    for k in path.split('.'):
        d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            if _NC is not None:
                logging.getLogger(__name__).warning(
                    'PRIME cuts JSON missing %r; code fallback %r in effect',
                    path, default)
            return default
    return d


# Nightly scan field scope: only candidates on these GB bulge fields are
# shown. Surfaced as the editable 'Fields' form input; blank -> this list.
PRIME_GB_FIELDS = [
    'GB130', 'GB109', 'GB131', 'GB113', 'GB111', 'GB80', 'GB63',
    'GB127', 'GB112', 'GB61', 'GB60', 'GB93', 'GB78', 'GB92',
    'GB57', 'GB97', 'GB91', 'GB94', 'GB75', 'GB76', 'GB110',
    'GB95', 'GB58', 'GB129', 'GB108', 'GB128', 'GB74', 'GB96',
    'GB62', 'GB114', 'GB126', 'GB59', 'GB125', 'GB79', 'GB77',
]


def _prime_parse_fieldlist(s):
    """Comma/whitespace-separated field names -> deduped uppercase list;
    blank -> PRIME_GB_FIELDS. Used for the field = ANY(...) cut."""
    toks = [t.strip().upper() for t in
            str(s or '').replace(',', ' ').split()]
    out, seen = [], set()
    for t in toks:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out or list(PRIME_GB_FIELDS)


_NIGHTLY_DEF = {
    'datemin': PRIME_DATE_LO, 'datemax': PRIME_DATE_HI,
    'fieldlist': ', '.join(PRIME_GB_FIELDS),
    'agelowlim': 0.0,
    'agehighlim': _nc('layer0.age_scope_days', 30.0),
    'scorrpeak': _nc('layer1.scorr_peak.lo', 7.09),
    'chi2hi': _nc('layer1.psf_chi2.hi', 271.23),
    'fwhmlo': _nc('layer1.fwhm.lo', 1.70),
    'fwhmhi': _nc('layer1.fwhm.hi', 5.25),
    'numneghi': _nc('layer1.numnegpix.hi', 24.4),
    'bwrefhi': _nc('layer1.numbadweightref.hi', 4.0),
    'bwscihi': _nc('layer1.numbadweightsci.hi', 9.0),
    'negclusthi': _nc('layer1.neglobe_clust.hi', 2.0),
    'negdistlo': _nc('layer1_gated.neglobe_dist_min', 3.6),
    'skipnum': 0,
}


def prime_nightly(form):
    """Layer 0 (date window + age scope on mjd - firstdet) + Layer 1 (shape
    cuts, incl. the neglobe_dist-gated-behind-clust>=3 conditional).
    Blank/missing form values fall back to the JSON-derived defaults."""
    def f(k):
        v = str(form.get(k, '') or '').strip()
        return float(v) if v != '' else float(_NIGHTLY_DEF[k])
    sql = ["filter2 = 'H' AND ispos = 1 AND mjd >= %s AND mjd < %s",
           'mjd - firstdet >= %s AND mjd - firstdet <= %s',        # layer 0
           'scorr_peak >= %s', 'psf_chi2 <= %s',                   # layer 1
           'fwhm >= %s AND fwhm <= %s',
           'numnegpix <= %s', 'numbadweightref <= %s',
           'numbadweightsci <= %s',
           'neglobe_clust <= %s',
           '(neglobe_clust < 3 OR neglobe_dist >= %s)']
    # Blank date fields fall back to the nightly default window
    # [today - 2 d, today], NOT full history; type an explicit early
    # datemin for a full-survey scan. Bare-MJD floats take the no-dash
    # branch of _prime_mjd_bound.
    params = [_prime_mjd_bound(form, 'datemin', Time.now().mjd - 2.0),
              _prime_mjd_bound(form, 'datemax', Time.now().iso[:10], end=True),
              f('agelowlim'), f('agehighlim'), f('scorrpeak'), f('chi2hi'),
              f('fwhmlo'), f('fwhmhi'), f('numneghi'), f('bwrefhi'),
              f('bwscihi'),
              f('negclusthi'), f('negdistlo')]
    # Field scope: candidates.field must be in the (form-editable) GB list.
    sql.append('field = ANY(%s)')
    params.append(_prime_parse_fieldlist(form.get('fieldlist', '')))
    return ' AND '.join(sql), params


_PF_NIGHTLY = [
    {'kind': 'daterange', 'label': 'Date (UTC)', 'lo': 'datemin', 'hi': 'datemax'},
    {'kind': 'range', 'label': 'Age (L0)', 'lo': 'agelowlim', 'hi': 'agehighlim',
     'suffix': 'days'},
    {'kind': 'num', 'label': 'Scorr peak \u2265', 'name': 'scorrpeak'},
    {'kind': 'num', 'label': 'psf_chi2 \u2264', 'name': 'chi2hi'},
    {'kind': 'range', 'label': 'fwhm', 'lo': 'fwhmlo', 'hi': 'fwhmhi'},
    {'kind': 'num', 'label': 'numnegpix \u2264', 'name': 'numneghi'},
    {'kind': 'num', 'label': 'bw ref \u2264', 'name': 'bwrefhi'},
    {'kind': 'num', 'label': 'bw sci \u2264', 'name': 'bwscihi'},
    {'kind': 'num', 'label': 'neglobe clust \u2264', 'name': 'negclusthi'},
    {'kind': 'num', 'label': '\u2026neglobe dist \u2265 (if clust\u22653)',
     'name': 'negdistlo'},
    {'kind': 'wide', 'label': 'Fields', 'name': 'fieldlist'},
]

PRIME.scans['nightly'] = {
    'template': 'prime_scan.html', 'builder': prime_nightly,
    'title': 'PRIME nightly transients (L0 age + L1 shape @T98)',
    'defaults': _NIGHTLY_DEF, 'fields': _PF_NIGHTLY,
}
# Adapter-skeleton stubs whose templates were never written: a POST to
# /prime/scan/{transient,spatial} raised TemplateNotFound -> 500.
PRIME.scans.pop('transient', None)
PRIME.scans.pop('spatial', None)
# <<< PRIME nightly scan

# >>> PRIME TOM upload
# Buttons on PRIME cards POST to /prime/tom_save (app.py). Config from env:
#   TOM_BASE_URL, TOM_TOKEN, PRIME_TOM_GROUPS (JSON: key -> {id, name[, list]}).
# Photometry: PRIME fp fluxes are counts @ ZP=24 Vega (make_prime_sources_fp);
# mag = 24 - 2.5 log10(flux), Vega. Detection := flux/staterr >= 5 (the fp
# convention used everywhere in the scanner); non-detections upload as limits
# at diffmaglim. The uploader runs in a background thread (a source can have
# ~500 fp rows x 1 POST each) and opens its OWN pooled cursor.
TOM_BASE_URL = os.environ.get('TOM_BASE_URL', '').rstrip('/')
TOM_TOKEN = os.environ.get('TOM_TOKEN', '')
try:
    PRIME_TOM_LISTS = _json.loads(os.environ.get('PRIME_TOM_LISTS', '{}'))
except Exception:
    PRIME_TOM_LISTS = {}
if not (TOM_BASE_URL and TOM_TOKEN and PRIME_TOM_LISTS):
    logging.getLogger(__name__).warning(
        'TOM upload not fully configured (TOM_BASE_URL/TOM_TOKEN/'
        'PRIME_TOM_LISTS); buttons will 400')

_TOM_HDRS = {'Authorization': 'Token %s' % TOM_TOKEN}
try:
    import urllib3; urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass
PRIME_TOM_ZP = 24.0
PRIME_TOM_SNR = 5.0


def prime_tom_get_or_create_target(name, ra, dec, group_key):
    """Return (target_id, created_bool); raises on TOM failure."""
    listname = PRIME_TOM_LISTS[group_key]
    r = requests.get('%s/api/targets/' % TOM_BASE_URL, params={'name': name},
                     headers=_TOM_HDRS, timeout=30, verify=False)
    if r.status_code == 200 and r.json().get('results'):
        return int(r.json()['results'][0]['id']), False
    _tom_gid = int(os.environ.get('PRIME_TOM_GROUP_ID', '3'))
    _tom_gname = os.environ.get('PRIME_TOM_GROUP_NAME', 'PRIME')
    r = requests.post('%s/api/targets/' % TOM_BASE_URL, headers=_TOM_HDRS,
                      timeout=30, verify=False,
                      json={'name': name, 'type': 'SIDEREAL',
                            'ra': float(ra), 'dec': float(dec),
                            'groups': [{'id': _tom_gid, 'name': _tom_gname}],
                            'target_lists': [{'name': listname}]})
    if r.status_code != 201:
        raise RuntimeError('TOM target create failed %d: %s'
                           % (r.status_code, r.text[:200]))
    return int(r.json()['id']), True


def prime_tom_upload_photometry(target_id, primeid):
    """All-band fp series -> TOM reduceddatums. Background thread; own cursor."""
    conn, cur = _prime_open()
    try:
        cur.execute('SELECT mjd, filter2, forcediffimflux, '
                    'forcediffimfluxstaterr, diffmaglim FROM forced_photometry '
                    'WHERE primeid = %s ORDER BY mjd ASC;', (primeid,))
        rows = cur.fetchall()
    finally:
        _prime_close(conn, cur)
    n_ok = n_fail = 0
    for o in rows:
        flux = o['forcediffimflux']; err = o['forcediffimfluxstaterr']
        if flux is None or err is None or err <= 0:
            continue
        band = 'PRIME_%s' % (o['filter2'] or 'NA')
        if flux > 0 and flux / err >= PRIME_TOM_SNR:
            value = {'magnitude': float(PRIME_TOM_ZP - 2.5 * np.log10(flux)),
                     'error': float(1.0857 * err / flux), 'filter': band}
        elif o['diffmaglim'] is not None:
            value = {'limit': float(o['diffmaglim']), 'filter': band}
        else:
            continue
        try:
            r = requests.post('%s/api/reduceddatums/' % TOM_BASE_URL,
                              headers=_TOM_HDRS, timeout=30, verify=False,
                              json={'target': target_id,
                                    'data_type': 'photometry',
                                    'timestamp': Time(o['mjd'], format='mjd').isot + 'Z',
                                    'value': value})
            n_ok += (r.status_code == 201)
            n_fail += (r.status_code != 201)
        except Exception:
            n_fail += 1
    logging.getLogger(__name__).info('TOM phot primeid=%s -> target %s: '
                                     '%d ok, %d failed', primeid, target_id,
                                     n_ok, n_fail)
# <<< PRIME TOM upload


# >>> PRIME per-source viewer (lookup by primeid or ra/dec) [prime_render_source_png]
# Standalone source page mirroring WTP search: resolve the sources row, render
# its forced-photometry LC directly from primeid (no candid needed) as
# cutouts + mag panel + full flux panel + a flux-space ZOOM around the peak.
# The generic /prime/cutout/<candid>.png path and _prime_render_png are
# untouched, so scan cards keep their existing figures.
PRIME_SOURCE_DEFAULTS = {'sourcename': '', 'rasearch': 268.0, 'decsearch': -29.0,
    'cmradius': 2.0}

PRIME_PEAK_ZOOM_DAYS = float(os.environ.get('PRIME_PEAK_ZOOM_DAYS', '60'))
PRIME_PEAK_ZOOM_MIN_DAYS = float(os.environ.get('PRIME_PEAK_ZOOM_MIN_DAYS', '15'))


def _prime_resolve_source(cur, form):
    # (primeid, ra, dec, name) or None. Exact source name wins; else nearest
    # sources row within cmradius arcsec of ra/dec. name is unique in sources.
    nm = str(form.get('sourcename', '') or '').strip()
    if nm:
        cur.execute('SELECT primeid, ra, dec, name FROM sources '
            'WHERE name = %s LIMIT 1;', (nm,))
        r = cur.fetchone()
        return (r['primeid'], r['ra'], r['dec'], r['name']) if r else None
    ra_s = str(form.get('rasearch', '') or '').strip()
    dec_s = str(form.get('decsearch', '') or '').strip()
    if ra_s == '' or dec_s == '':
        return None
    ra, dec = float(ra_s), float(dec_s)
    rad = float(form.get('cmradius', 2.0) or 2.0) / 3600.0
    cur.execute('SELECT primeid, ra, dec, name FROM sources '
        'WHERE q3c_radial_query(ra, dec, %s, %s, %s) '
        'ORDER BY q3c_dist(ra, dec, %s, %s) ASC LIMIT 1;',
        (ra, dec, rad, ra, dec))
    r = cur.fetchone()
    return (r['primeid'], r['ra'], r['dec'], r['name']) if r else None


def _prime_fmt(v, fmt='%s'):
    if v is None:
        return '\u2014'
    try:
        return fmt % v
    except Exception:
        return str(v)


def build_prime_source_context(survey, form):
    conn, cur = survey.open_cursor()
    try:
        resolved = _prime_resolve_source(cur, form)
        if resolved is None:
            return 'prime_show_source.html', dict(
                notfound=True, source=None, meta=[], coords_sex='',
                scan_query='', tom_lists=PRIME_TOM_LISTS,
                defpardict=form.to_dict())
        primeid, ra, dec, name = resolved
        cols = ', '.join('cand.%s' % c for _, c in survey.field_map)
        # Peak candidate (highest scorr_peak within the astrometric match
        # radius) used only for the metadata table; the LC image is rendered
        # straight from primeid, so a source with no positive candidate still
        # shows its forced curve.
        sql = ('SELECT %s FROM candidates cand '
            'WHERE q3c_radial_query(cand.ra, cand.dec, %%s, %%s, %%s) '
            'AND ispos = 1 ORDER BY cand.scorr_peak DESC LIMIT 1;' % cols)
        qparams = (ra, dec, PRIME_CMRADIUS)
        qstr = cur.mogrify(sql, qparams).decode()
        cur.execute(sql, qparams)
        peak = cur.fetchone()
    finally:
        survey.close_cursor(conn, cur)

    meta = []
    if peak is not None:
        g = peak.get
        age = None
        if g('mjd') is not None and g('firstdet') is not None:
            age = g('mjd') - g('firstdet')
        meta = [
            ('Peak candid', _prime_fmt(g('candid'))),
            ('Peak MJD', _prime_fmt(g('mjd'), '%.4f')),
            ('First det MJD', _prime_fmt(g('firstdet'), '%.4f')),
            ('Age at peak (d)', _prime_fmt(age, '%.1f')),
            ('Filter', _prime_fmt(g('filter2'))),
            ('Field', _prime_fmt(g('field'))),
            ('fpapos', _prime_fmt(g('fpapos'))),
            ('scorr_peak', _prime_fmt(g('scorr_peak'), '%.2f')),
            ('psf_mag', _prime_fmt(g('psf_mag'), '%.3f')),
            ('rbscore', _prime_fmt(g('rbscore'), '%.3f')),
            ('fwhm', _prime_fmt(g('fwhm'), '%.2f')),
            ('psf_chi2', _prime_fmt(g('psf_chi2'), '%.2f')),
            ('numnegpix', _prime_fmt(g('numnegpix'))),
            ('neglobe_clust', _prime_fmt(g('neglobe_clust'))),
            ('neglobe_dist', _prime_fmt(g('neglobe_dist'), '%.2f')),
            ('VVV dist1 (")', _prime_fmt(g('vvvdist1'), '%.2f')),
            ('VVV H1', _prime_fmt(g('vvvmagh1'), '%.2f')),
            ('2MASS dist1 (")', _prime_fmt(g('tmdist1'), '%.2f')),
            ('2MASS H1', _prime_fmt(g('tmmagh1'), '%.2f')),
            ('dist nearbrstar (")', _prime_fmt(g('distnearbrstar'), '%.2f')),
        ]

    try:
        coords_sex = SkyCoord(ra=ra, dec=dec, unit='degree',
            frame='icrs').to_string('hmsdms', precision=2)
    except Exception:
        coords_sex = ''

    return 'prime_show_source.html', dict(
        notfound=False,
        source=dict(primeid=primeid, ra=float(ra), dec=float(dec), name=name),
        meta=meta, coords_sex=coords_sex, scan_query=qstr,
        tom_lists=PRIME_TOM_LISTS, defpardict=form.to_dict())


def prime_render_source_png(survey, primeid):
    # Source-level LC straight from primeid: cutouts (by sourcename) + mag +
    # full flux + a flux-space zoom around the peak. Reuses the PRIME fp
    # conventions (SNR floor, mag floor, per-band staterr outlier mask).
    import io as _io
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.gridspec import GridSpec
    from astropy.time import Time as _Time
    from astropy.stats import sigma_clipped_stats as _scs

    conn, cur = survey.open_cursor()
    try:
        cur.execute('SELECT name, ra, dec FROM sources WHERE primeid = %s LIMIT 1;',
            (primeid,))
        src = cur.fetchone()
        if src is None:
            raise KeyError(primeid)
        name = src['name']
        cur.execute('SELECT candid, mjd, sci_image, ref_image, diff_image '
            'FROM cutouts WHERE sourcename = %s;', (name,))
        c = cur.fetchone()
        cur.execute('SELECT mjd, filter2, forcediffimflux, forcediffimfluxstaterr, '
            'forcediffmagpsf, forcediffsigmapsf, diffmaglim '
            'FROM forced_photometry WHERE primeid = %s ORDER BY mjd ASC;', (primeid,))
        out = cur.fetchall()
    finally:
        survey.close_cursor(conn, cur)

    def _loadimg(blob):
        b = bytes(blob)
        if b[:2] == b'\x1f\x8b':
            b = gzip.open(io.BytesIO(b), 'rb').read()
        return np.flipud(fits.open(io.BytesIO(b))[0].data)
    cut = None
    if c is not None:
        cut = {'sci': _loadimg(c['sci_image']), 'ref': _loadimg(c['ref_image']),
               'diff': _loadimg(c['diff_image']), 'mjd': c['mjd']}

    if cut is not None:
        fig = Figure(figsize=(10, 12))
        gs = GridSpec(4, 3, figure=fig, height_ratios=[1.0, 1.1, 1.1, 1.1],
            hspace=0.5, wspace=0.05)
        axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
        ax_mag = fig.add_subplot(gs[1, :])
        ax_flx = fig.add_subplot(gs[2, :], sharex=ax_mag)
        ax_zoom = fig.add_subplot(gs[3, :])
        for a, img, ttl in zip(axes, (cut['sci'], cut['ref'], cut['diff']),
                ('Science', 'Reference', 'Difference')):
            _, med, std = _scs(img)
            a.imshow(img, cmap='gray', vmin=med - std, vmax=med + 5 * std,
                aspect='equal')
            a.set_title(ttl, fontsize=14)
            a.set_xticks([]); a.set_yticks([])
    else:
        fig = Figure(figsize=(10, 9))
        gs = GridSpec(3, 1, figure=fig, hspace=0.4)
        ax_mag = fig.add_subplot(gs[0])
        ax_flx = fig.add_subplot(gs[1], sharex=ax_mag)
        ax_zoom = fig.add_subplot(gs[2])

    zlo = zhi = peak_mjd = None
    if out:
        mjd_a = _farr(out, 'mjd')
        bid_a = np.array([PRIME_FILTER_BAND.get(o['filter2'], 0) for o in out])
        flux = _farr(out, 'forcediffimflux')
        ferr = _farr(out, 'forcediffimfluxstaterr')
        mag = _farr(out, 'forcediffmagpsf')
        magerr = _farr(out, 'forcediffsigmapsf')
        lim = _farr(out, 'diffmaglim')

        keep = _prime_band_outlier_mask(bid_a, ferr)
        with np.errstate(invalid='ignore', divide='ignore'):
            snr = np.where(ferr > 0, flux / ferr, np.nan)
        isdet = (np.isfinite(snr) & (snr >= PRIME_FP_SNR)
            & np.isfinite(mag) & (mag > 0))

        # mag panel (floor on plotted value, like prime_fetch_forced)
        shown = np.where(isdet, mag, lim)
        mag_keep = keep & ~(np.isfinite(shown) & (shown > PRIME_FP_MAGFLOOR))
        for b in survey.bands:
            sel = mag_keep & (bid_a == b.id)
            if not np.any(sel):
                continue
            det = sel & isdet
            lm = sel & ~isdet & np.isfinite(lim)
            if np.any(det):
                ax_mag.errorbar(mjd_a[det], mag[det], yerr=np.abs(magerr[det]),
                    ls='none', marker='o', color=b.color, ms=9, label=b.label)
            if np.any(lm):
                ax_mag.errorbar(mjd_a[lm], lim[lm], yerr=0.2, uplims=True,
                    ls='none', marker='v', color=b.color,
                    markerfacecolor='none', ms=7)

        # full + zoom flux panels (signed, all kept epochs)
        for b in survey.bands:
            sel = keep & (bid_a == b.id)
            if not np.any(sel):
                continue
            ax_flx.errorbar(mjd_a[sel], flux[sel], yerr=ferr[sel], ls='none',
                marker='o', color=b.color, ms=6, label=b.label)
            ax_zoom.errorbar(mjd_a[sel], flux[sel], yerr=ferr[sel], ls='none',
                marker='o', color=b.color, ms=7, label=b.label)
        ax_flx.axhline(0.0, color='0.5', lw=0.8)
        ax_zoom.axhline(0.0, color='0.5', lw=0.8)

        # peak in flux space: brightest kept detection (fall back to any kept
        # finite flux). Window adapts to the >=20%-of-peak detection span.
        pk = keep & isdet & np.isfinite(flux)
        if not pk.any():
            pk = keep & np.isfinite(flux)
        if pk.any():
            fmax = np.where(pk, flux, -np.inf)
            pi = int(np.argmax(fmax))
            peak_mjd = float(mjd_a[pi]); peak_flux = float(flux[pi])
            hi = pk & (flux >= 0.2 * peak_flux)
            if np.count_nonzero(hi) >= 2 and peak_flux > 0:
                span = float(mjd_a[hi].max() - mjd_a[hi].min())
                half = max(0.75 * span, PRIME_PEAK_ZOOM_MIN_DAYS)
            else:
                half = PRIME_PEAK_ZOOM_DAYS
            zlo, zhi = peak_mjd - half, peak_mjd + half
            # y-zoom to the points inside the window
            win = keep & np.isfinite(flux) & (mjd_a >= zlo) & (mjd_a <= zhi)
            if win.any():
                ylo = float(np.nanmin((flux - ferr)[win]))
                yhi = float(np.nanmax((flux + ferr)[win]))
                pad = 0.08 * (yhi - ylo if yhi > ylo else abs(yhi) + 1.0)
                ax_zoom.set_ylim(ylo - pad, yhi + pad)

    ax_mag.invert_yaxis()
    ax_mag.set_ylabel('Mag', fontsize=13)
    ax_mag.tick_params(labelsize=11)
    ax_mag.set_title('PRIME %d; %s' % (primeid, name), fontsize=14)
    if ax_mag.get_legend_handles_labels()[1]:
        ax_mag.legend(fontsize=9)
    ax_flx.set_ylabel('Flux (diff)', fontsize=13)
    ax_flx.set_xlabel('MJD', fontsize=13)
    ax_flx.tick_params(labelsize=11)
    ax_zoom.set_ylabel('Flux (diff)', fontsize=13)
    ax_zoom.set_xlabel('MJD (peak zoom)', fontsize=13)
    ax_zoom.tick_params(labelsize=11)

    if peak_mjd is not None:
        for axx in (ax_mag, ax_flx, ax_zoom):
            axx.axvline(peak_mjd, color='g', ls=':', lw=1.0, alpha=0.7)
    if zlo is not None:
        ax_zoom.set_xlim(zlo, zhi)
        ax_zoom.set_title('Flux zoom @ peak (MJD %.1f \u00b1 %.0f d)'
            % (peak_mjd, (zhi - zlo) / 2.0), fontsize=12)
    else:
        ax_zoom.set_title('Flux zoom @ peak (no detection)', fontsize=12)

    if cut is not None and cut.get('mjd') is not None \
            and np.isfinite(float(cut['mjd'])):
        for axx in (ax_mag, ax_flx):
            axx.axvline(float(cut['mjd']), color='r', ls='--', lw=1.0, alpha=0.6)

    if out:
        ax_yr = ax_mag.twiny()
        ax_yr.set_xlim(_Time(ax_mag.get_xlim(), format='mjd').decimalyear)
        ax_yr.set_xlabel('Year', fontsize=11)
        ax_yr.tick_params(labelsize=10)

    fig.tight_layout()
    FigureCanvasAgg(fig)
    buf = _io.BytesIO()
    fig.savefig(buf, format='png', dpi=80)
    return buf.getvalue()
# <<< PRIME per-source viewer


# >>> PRIME source viewer: interactive LC data + cutouts strip [prime_source_lc_data]
# JSON series for the client-side (Plotly) light curve, plus a cutouts-only PNG.
# Reuses the PRIME fp conventions (SNR floor, mag floor, per-band staterr
# outlier mask) and the same peak / zoom-window logic as the static renderer.
# All non-finite values are emitted as null so response.json() never chokes on
# a bare NaN token.
PRIME_WEB_BANDCOLOR = {'Y': '#e8830c', 'J': '#c02bd3', 'H': '#1f77b4',
                       'Z': '#7a3ff2'}


def _prime_web_color(band):
    return PRIME_WEB_BANDCOLOR.get(band.label, '#1f77b4')


def _jl(a):
    """np array -> JSON list with None for non-finite / None."""
    out = []
    for x in a:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            out.append(None); continue
        out.append(xf if np.isfinite(xf) else None)
    return out


def prime_source_lc_data(survey, primeid):
    conn, cur = survey.open_cursor()
    try:
        cur.execute('SELECT name, ra, dec FROM sources WHERE primeid = %s LIMIT 1;',
            (primeid,))
        src = cur.fetchone()
        if src is None:
            raise KeyError(primeid)
        cur.execute('SELECT mjd, filter2, forcediffimflux, forcediffimfluxstaterr, '
            'forcediffmagpsf, forcediffsigmapsf, diffmaglim '
            'FROM forced_photometry WHERE primeid = %s ORDER BY mjd ASC;',
            (primeid,))
        out = cur.fetchall()
    finally:
        survey.close_cursor(conn, cur)

    payload = {'primeid': int(primeid), 'name': src['name'],
        'ra': float(src['ra']), 'dec': float(src['dec']),
        'peak_mjd': None, 'zoom_lo': None, 'zoom_hi': None, 'bands': []}
    if not out:
        return payload

    mjd_a = _farr(out, 'mjd')
    bid_a = np.array([PRIME_FILTER_BAND.get(o['filter2'], 0) for o in out])
    flux = _farr(out, 'forcediffimflux')
    ferr = _farr(out, 'forcediffimfluxstaterr')
    mag = _farr(out, 'forcediffmagpsf')
    magerr = _farr(out, 'forcediffsigmapsf')
    lim = _farr(out, 'diffmaglim')

    keep = _prime_band_outlier_mask(bid_a, ferr)
    with np.errstate(invalid='ignore', divide='ignore'):
        snr = np.where(ferr > 0, flux / ferr, np.nan)
    isdet = np.isfinite(snr) & (snr >= PRIME_FP_SNR) & np.isfinite(mag) & (mag > 0)
    shown = np.where(isdet, mag, lim)
    mag_keep = keep & ~(np.isfinite(shown) & (shown > PRIME_FP_MAGFLOOR))

    for b in survey.bands:
        band = (bid_a == b.id)
        det = mag_keep & band & isdet
        lm = mag_keep & band & ~isdet & np.isfinite(lim)
        fx = keep & band
        if not (det.any() or lm.any() or fx.any()):
            continue
        payload['bands'].append({
            'label': b.label, 'color': _prime_web_color(b),
            'det': {'mjd': _jl(mjd_a[det]), 'mag': _jl(mag[det]),
                    'magerr': _jl(np.abs(magerr[det])),
                    'flux': _jl(flux[det]), 'ferr': _jl(ferr[det])},
            'lim': {'mjd': _jl(mjd_a[lm]), 'mag': _jl(lim[lm])},
            'flux_all': {'mjd': _jl(mjd_a[fx]), 'val': _jl(flux[fx]),
                         'err': _jl(ferr[fx])},
        })

    # peak (brightest kept detection in flux; fall back to any kept flux) and
    # an adaptive zoom window (0.75x the >=20%-of-peak detection span, floored).
    pk = keep & isdet & np.isfinite(flux)
    if not pk.any():
        pk = keep & np.isfinite(flux)
    if pk.any():
        pi = int(np.argmax(np.where(pk, flux, -np.inf)))
        peak_mjd = float(mjd_a[pi]); peak_flux = float(flux[pi])
        hi = pk & (flux >= 0.2 * peak_flux)
        if np.count_nonzero(hi) >= 2 and peak_flux > 0:
            span = float(mjd_a[hi].max() - mjd_a[hi].min())
            half = max(0.75 * span, PRIME_PEAK_ZOOM_MIN_DAYS)
        else:
            half = PRIME_PEAK_ZOOM_DAYS
        payload['peak_mjd'] = peak_mjd
        payload['zoom_lo'] = peak_mjd - half
        payload['zoom_hi'] = peak_mjd + half
    return payload


def prime_render_source_cutouts_png(survey, primeid):
    """3-panel sci/ref/diff strip for the source (by sourcename), for the
    interactive page header. Raises KeyError if the source or its cutout row
    is absent."""
    import io as _io
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from astropy.stats import sigma_clipped_stats as _scs

    conn, cur = survey.open_cursor()
    try:
        cur.execute('SELECT name FROM sources WHERE primeid = %s LIMIT 1;', (primeid,))
        s = cur.fetchone()
        if s is None:
            raise KeyError(primeid)
        cur.execute('SELECT sci_image, ref_image, diff_image FROM cutouts '
            'WHERE sourcename = %s;', (s['name'],))
        c = cur.fetchone()
    finally:
        survey.close_cursor(conn, cur)
    if c is None:
        raise KeyError(primeid)

    def _loadimg(blob):
        b = bytes(blob)
        if b[:2] == b'\x1f\x8b':
            b = gzip.open(io.BytesIO(b), 'rb').read()
        return np.flipud(fits.open(io.BytesIO(b))[0].data)

    fig = Figure(figsize=(9, 3.2))
    for k, (img, ttl) in enumerate(zip(
            (_loadimg(c['sci_image']), _loadimg(c['ref_image']),
             _loadimg(c['diff_image'])), ('Science', 'Reference', 'Difference'))):
        a = fig.add_subplot(1, 3, k + 1)
        _, med, std = _scs(img)
        a.imshow(img, cmap='gray', vmin=med - std, vmax=med + 5 * std, aspect='equal')
        a.set_title(ttl, fontsize=13)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    FigureCanvasAgg(fig)
    buf = _io.BytesIO()
    fig.savefig(buf, format='png', dpi=90)
    return buf.getvalue()


def prime_source_fp_csv(survey, primeid):
    """Full forced-photometry table for a source (by primeid), as CSV text.
    No outlier / mag-floor cuts -- the complete raw forced_photometry record,
    all bands, all epochs, plus derived snr/isdet. Returns (name, csv_text);
    KeyError if the source is absent."""
    import csv as _csv
    import io as _io
    conn, cur = survey.open_cursor()
    try:
        cur.execute('SELECT name, ra, dec FROM sources WHERE primeid = %s LIMIT 1;',
            (primeid,))
        src = cur.fetchone()
        if src is None:
            raise KeyError(primeid)
        cur.execute('SELECT mjd, filter2, forcediffimflux, forcediffimfluxstaterr, '
            'forcediffmagpsf, forcediffsigmapsf, diffmaglim '
            'FROM forced_photometry WHERE primeid = %s ORDER BY mjd ASC;',
            (primeid,))
        out = cur.fetchall()
    finally:
        survey.close_cursor(conn, cur)
    buf = _io.StringIO()
    buf.write('# PRIME forced photometry\n')
    buf.write('# primeid=%d  name=%s  ra=%.6f  dec=%.6f\n'
        % (int(primeid), src['name'], float(src['ra']), float(src['dec'])))
    buf.write('# mag = 24 - 2.5*log10(flux) (Vega); detection = '
        'flux/staterr >= %.1f; no outlier/floor cuts applied\n' % PRIME_FP_SNR)
    w = _csv.writer(buf)
    w.writerow(['mjd', 'filter', 'diffflux', 'difffluxstaterr', 'diffmagpsf',
        'diffsigmapsf', 'diffmaglim', 'snr', 'isdet'])
    for o in out:
        flux, err = o['forcediffimflux'], o['forcediffimfluxstaterr']
        if flux is not None and err not in (None, 0):
            snr = flux / err
            snr_s, isdet = '%.4f' % snr, int(snr >= PRIME_FP_SNR)
        else:
            snr_s, isdet = '', ''
        w.writerow([o['mjd'], o['filter2'], flux, err, o['forcediffmagpsf'],
            o['forcediffsigmapsf'], o['diffmaglim'], snr_s, isdet])
    return src['name'], buf.getvalue()

# <<< PRIME source viewer interactive

# >>> PRIME badges: TOM membership + MOA-PRIME known-alert crossmatch
# (i) TOM: exact-name lookup, cached per name (TTL, stale-on-failure). Uses the
# same TOM_BASE_URL/TOM_TOKEN/_TOM_HDRS as the uploader.
# (ii) MOA: nearest scraped alert within PRIME_ALERT_RADIUS arcsec. Positions
# come from prime_*.csv produced by scrape_prime_alerts.py (real display-page
# ra/dec). Files are globbed and cached, invalidated on any file's mtime, so a
# re-scrape refreshes the badge without a restart. Blank-position rows skipped.
import glob as _glob
import csv as _csv_badge

PRIME_ALERTS_GLOB = os.environ.get('PRIME_ALERTS_CSV',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prime_*.csv'))
PRIME_ALERT_RADIUS_DEG = float(
    os.environ.get('PRIME_ALERT_RADIUS_ARCSEC', '2.0')) / 3600.0
PRIME_TOM_NAME_TTL = float(os.environ.get('PRIME_TOM_NAME_TTL', '300'))

_TOM_NAME_CACHE = {}                 # name -> (ts, bool)
_TOM_NAME_LOCK = threading.Lock()
_ALERTS = {'ra': None, 'dec': None, 'names': None, 'mtime': None, 'paths': ()}
_ALERTS_LOCK = threading.Lock()


def prime_tom_has_name(name):
    """TOM target id (int) for this exact name, or None. Cached per name."""
    if not name or name == 'NULL':
        return None
    try:
        base, tok, hdrs = TOM_BASE_URL, TOM_TOKEN, _TOM_HDRS
    except NameError:
        return None
    if not (base and tok):
        return None
    now = time.time()
    with _TOM_NAME_LOCK:
        hit = _TOM_NAME_CACHE.get(name)
        if hit and now - hit[0] < PRIME_TOM_NAME_TTL:
            return hit[1]
    try:
        r = requests.get('%s/api/targets/' % base, params={'name': name},
                         headers=hdrs, timeout=15, verify=False)
        tid = None
        if r.status_code == 200:
            res = r.json().get('results') or []
            if res:
                try:
                    tid = int(res[0]['id'])
                except (KeyError, TypeError, ValueError):
                    tid = None
    except Exception:
        with _TOM_NAME_LOCK:                 # stale-on-failure
            hit = _TOM_NAME_CACHE.get(name)
        return hit[1] if hit else None
    with _TOM_NAME_LOCK:
        _TOM_NAME_CACHE[name] = (now, tid)
    return tid


def _load_alerts():
    """(Re)load scraped MOA-PRIME positions from prime_*.csv into _ALERTS.
    Cached on the set of files + max mtime; ra/dec parsed once to deg. Accepts
    decimal degrees or HMS/DMS (hourangle/deg), matching the scraper output."""
    import astropy.units as u
    paths = tuple(sorted(_glob.glob(PRIME_ALERTS_GLOB)))
    mt = max((os.path.getmtime(p) for p in paths), default=None)
    with _ALERTS_LOCK:
        if (_ALERTS['ra'] is not None and _ALERTS['paths'] == paths
                and _ALERTS['mtime'] == mt):
            return
        ras, decs, names = [], [], []
        for p in paths:
            try:
                with open(p, newline='', encoding='utf-8') as fh:
                    for row in _csv_badge.DictReader(fh):
                        rs = (row.get('ra') or '').strip()
                        ds = (row.get('dec') or '').strip()
                        if not rs or not ds:
                            continue            # blank -> not assumed
                        try:
                            ra, dec = float(rs), float(ds)   # decimal deg
                        except ValueError:
                            try:
                                c = SkyCoord(rs, ds, unit=(u.hourangle, u.deg))
                                ra, dec = c.ra.deg, c.dec.deg
                            except Exception:
                                continue
                        ras.append(ra); decs.append(dec)
                        names.append(row.get('name', ''))
            except Exception:
                continue
        _ALERTS['ra'] = np.asarray(ras, dtype=float)
        _ALERTS['dec'] = np.asarray(decs, dtype=float)
        _ALERTS['names'] = np.asarray(names, dtype=object)
        _ALERTS['mtime'] = mt
        _ALERTS['paths'] = paths


def prime_alert_match_array(ras, decs):
    """Array of matched MOA alert names (nearest within the match radius), ''
    where no alert falls inside the radius."""
    _load_alerts()
    n = len(ras)
    out = np.array([''] * n, dtype=object)
    ar, ad = _ALERTS['ra'], _ALERTS['dec']
    nm = _ALERTS['names']
    if ar is None or len(ar) == 0:
        return out
    ar_r, ad_r = np.radians(ar), np.radians(ad)
    thr = np.radians(PRIME_ALERT_RADIUS_DEG)
    sin_ad, cos_ad = np.sin(ad_r), np.cos(ad_r)
    for i in range(n):
        try:
            ra0 = np.radians(float(ras[i])); dec0 = np.radians(float(decs[i]))
        except (TypeError, ValueError):
            continue
        sep = np.arccos(np.clip(
            np.sin(dec0) * sin_ad + np.cos(dec0) * cos_ad * np.cos(ar_r - ra0),
            -1.0, 1.0))
        j = int(np.argmin(sep))
        if sep[j] < thr:
            out[i] = str(nm[j]) if nm is not None and j < len(nm) else ''
    return out


def prime_alert_match_one(ra, dec):
    m = prime_alert_match_array([ra], [dec])
    return m[0] if len(m) else ''


def _prime_tom_ids_parallel(names, workers=8, deadline=20.0):
    """prime_tom_has_name over unique names, concurrently, under one
    overall deadline. The old serial loop was O(n) HTTP round-trips at
    15 s timeout each on a cold cache -- one slow TOM stalled the whole
    scan render. Unresolved names come back None (no badge), never an
    exception; the per-name TTL cache still applies underneath."""
    from concurrent.futures import ThreadPoolExecutor
    uniq = sorted({nm for nm in names if nm and nm != 'NULL'})
    resolved = {}
    if uniq:
        ex = ThreadPoolExecutor(max_workers=min(workers, len(uniq)))
        futs = {nm: ex.submit(prime_tom_has_name, nm) for nm in uniq}
        t_end = time.time() + deadline
        for nm, fu in futs.items():
            try:
                resolved[nm] = fu.result(
                    timeout=max(0.0, t_end - time.time()))
            except Exception:
                resolved[nm] = None
        ex.shutdown(wait=False, cancel_futures=True)
    return [resolved.get(nm) for nm in names]


# Chain onto the existing (sex-coords) build_scan_context so PRIME scan cards
# gain tom_ex / alert_ex arrays aligned with candids.
_prev_bsc_badges = build_scan_context
def build_scan_context(survey, scan_name, form):
    template, ctx = _prev_bsc_badges(survey, scan_name, form)
    try:
        if getattr(survey, 'key', None) == 'prime':
            cd = ctx.get('canddicts')
            if cd and len(cd.get('ras', [])):
                names = list(cd.get('names', []))
                tids = _prime_tom_ids_parallel(names)
                cd['tom_ids'] = np.array(tids, dtype=object)
                cd['tom_ex'] = np.array([t is not None for t in tids], dtype=bool)
                anames = prime_alert_match_array(cd['ras'], cd['decs'])
                cd['alert_names'] = anames
                cd['alert_ex'] = np.array([bool(a) for a in anames], dtype=bool)
                ctx['tom_web_base'] = (TOM_BASE_URL or '')
                ctx['moa_display_base'] = "https://moaprime.massey.ac.nz/alerts/display/"
    except Exception as _e:
        logging.getLogger(__name__).warning('prime badge annotate failed: %s', _e)
    return template, ctx


# Same for the single-source page context.
_prev_bpsc_badges = build_prime_source_context
def build_prime_source_context(survey, form):
    template, ctx = _prev_bpsc_badges(survey, form)
    try:
        src = ctx.get('source')
        if src:
            tid = prime_tom_has_name(src.get('name'))
            ctx['tom_id'] = tid
            ctx['tom_ex'] = tid is not None
            aname = prime_alert_match_one(src['ra'], src['dec'])
            ctx['alert_name'] = aname
            ctx['alert_ex'] = bool(aname)
        else:
            ctx['tom_id'] = None; ctx['alert_name'] = ''
            ctx['tom_ex'] = ctx['alert_ex'] = False
        ctx['tom_web_base'] = (TOM_BASE_URL or '')
        ctx['moa_display_base'] = "https://moaprime.massey.ac.nz/alerts/display/"
    except Exception as _e:
        logging.getLogger(__name__).warning('prime source badge failed: %s', _e)
        ctx['tom_id'] = None; ctx['alert_name'] = ''
        ctx['tom_ex'] = ctx['alert_ex'] = False
    return template, ctx
# <<< PRIME badges


# [patch] wtp-bugfix-20260717 applied

# [patch] wtp-nightly-window-20260717 applied

# [patch] wtp-gb-fields-20260717 applied


# >>> NEOWISE x ZTF/LSST crossmatch scan (SKELETON -- fill the *** markers)
# ---------------------------------------------------------------------------
# New scanning tab: NEOWISE transients crossmatched to ZTF and/or LSST.
#
# Design (mirrors the PRIME patch pattern above):
#   1. WTPX = a pseudo-survey sharing the WTP (NEOWISE) connection + schema,
#      registered as key 'wtpx'. Registering a separate key (rather than
#      adding a scan to WTP) gives the scan its OWN cutout URL namespace
#      (/wtpx/cutout/<candid>.png), which is how the renderer knows to draw
#      the extra ZTF/LSST panel -- the cutout route only knows (survey, candid).
#   2. One scan 'xmatch': the hostless-tab criteria (_wtp_base + hlwdist; swap
#      in any other builder if you'd rather clone the hosted/nuclear tab),
#      INNER JOINed to source_fp_stats directly on candid, requiring
#      ztfname/lsstname per the form.
#   3. A wtpx-only renderer: standard NEOWISE cutouts + LC figure, plus a
#      second LC panel (shared MJD axis) with ZTF/LSST photometry pulled from
#      the Babamul alert broker (TTL-cached, never fails the render).
#
# Routes come for free from app.py's generic /<survey_key>/scan/<scan_name>
# route; the only app-side change needed is a menu link (see wise_scan.html).
# ---------------------------------------------------------------------------
import dataclasses as _dc

# --- Babamul broker client --------------------------------------------------
# Uses the babamul package (get_photometry), auth'd via BABAMUL_API_TOKEN /
# BABAMUL_ENV per babamul.config -- no ad-hoc REST client needed. Cache:
# (survey, name) -> (ts, lc|None), TTL'd; on any failure return the stale
# entry or None -- a scan render must never fail because the broker is down
# (same rule as the Fritz cache above).
from babamul.api import get_photometry
from babamul.lightcurves import SNR_THRESHOLD, _normalize_band
from babamul.exceptions import BabamulError

BABAMUL_TTL = float(os.environ.get('BABAMUL_TTL', '3600'))
_BABAMUL_CACHE = {}
_BABAMUL_LOCK = threading.Lock()

def _phot_isdiffpos(p):
    """isdiffpos, falling back to the psfFlux sign when the package leaves it
    unset. babamul.raw_models.Photometry.from_forced_photometry (used to
    build fp_hists rows) never sets isdiffpos -- it stays at the field
    default None regardless of the flux sign -- while from_alert_photometry
    (prv_candidates) does set it correctly. Without this fallback, every
    forced-photometry detection renders as if it were positive."""
    if p.isdiffpos is not None:
        return p.isdiffpos
    if p.psfFlux is not None:
        return p.psfFlux > 0
    return None

def _babamul_flatten(prv_candidates, prv_nondetections, fp_hists):
    """Flatten babamul Photometry records into per-point dicts, mirroring
    babamul.lightcurves.get_prv_candidates/get_prv_nondetections/get_fp_hists
    (same SNR_THRESHOLD detection/limit split) but keeping isdiffpos, which
    those helpers drop -- needed to mark negative-flux ('isdiffpos=False')
    detections with a distinct marker."""
    rows = []
    for p in prv_candidates + fp_hists:
        mjd = p.jd - 2400000.5
        band = _normalize_band(p.band)
        if p.snr and p.snr >= SNR_THRESHOLD:
            rows.append({'mjd': mjd, 'mag': p.magpsf, 'magerr': p.sigmapsf,
                'band': band, 'lim': False, 'isdiffpos': _phot_isdiffpos(p)})
        elif p.diffmaglim is not None:
            rows.append({'mjd': mjd, 'mag': p.diffmaglim, 'magerr': None,
                'band': band, 'lim': True, 'isdiffpos': None})
    for p in prv_nondetections:
        rows.append({'mjd': p.jd - 2400000.5, 'mag': p.diffmaglim,
            'magerr': None, 'band': _normalize_band(p.band), 'lim': True,
            'isdiffpos': None})
    return rows

def babamul_get_lightcurve(ext_survey, name):
    """Fetch a ZTF or LSST light curve for `name` from Babamul.

    ext_survey: 'ztf' | 'lsst'.  Returns a normalized dict of np arrays
    {'mjd', 'mag', 'magerr', 'limmag', 'isdet', 'isneg', 'filt'} (filt =
    filter-name strings, e.g. 'g','r','i' / 'u'..'y'; isneg = detection with
    isdiffpos == False, i.e. a negative-flux/subtraction-artifact point), or
    None when unavailable.
    """
    key = (ext_survey, name)
    now = time.time()
    with _BABAMUL_LOCK:
        hit = _BABAMUL_CACHE.get(key)
        if hit and now - hit[0] < BABAMUL_TTL:
            return hit[1]
    try:
        phot = get_photometry(ext_survey.upper(), name)
        rows = _babamul_flatten(phot.prv_candidates, phot.prv_nondetections,
            phot.fp_hists)
        if not rows:
            lc = None
        else:
            isdet = np.array([not r['lim'] for r in rows], dtype=bool)
            mag = np.array([np.nan if r['mag'] is None else r['mag']
                for r in rows], dtype=float)
            lc = {
                'mjd': np.array([r['mjd'] for r in rows], dtype=float),
                'mag': np.where(isdet, mag, np.nan),
                'magerr': np.array([np.nan if r['magerr'] is None else r['magerr']
                    for r in rows], dtype=float),
                'limmag': np.where(isdet, np.nan, mag),
                'isdet': isdet,
                'isneg': np.array([r['isdiffpos'] is False for r in rows], dtype=bool),
                'filt': np.array([_normalize_band(r['band']) for r in rows]),
            }
    except BabamulError as e:
        logging.getLogger(__name__).warning('babamul %s/%s failed: %s',
            ext_survey, name, e)
        with _BABAMUL_LOCK:
            hit = _BABAMUL_CACHE.get(key)
        return hit[1] if hit else None              # stale beats broken
    with _BABAMUL_LOCK:
        _BABAMUL_CACHE[key] = (now, lc)
    return lc

# Colors per (survey, filter); marker distinguishes survey on the plot
# (ZTF = circles, LSST = squares). *** confirm LSST filter naming from broker.
EXT_BAND_COLORS = {
    ('ztf', 'g'): 'green', ('ztf', 'r'): 'red', ('ztf', 'i'): 'orange',
    ('lsst', 'u'): 'purple', ('lsst', 'g'): 'g', ('lsst', 'r'): 'r',
    ('lsst', 'i'): 'goldenrod', ('lsst', 'z'): 'brown', ('lsst', 'y'): '0.4',
}
EXT_MARKERS = {'ztf': 'o', 'lsst': 's'}

# --- Crossmatch join + predicate builder ------------------------------------
# INNER JOIN source_fp_stats directly on candid -- confirmed join key. INNER
# means candidates with no source_fp_stats row (not yet xmatched) drop out,
# which is the point of this tab.
WTPX_JOIN = 'INNER JOIN source_fp_stats xm ON xm.candid = cand.candid'

# Surfaced onto the result cards by serialize_candidates (extra_select cols
# are auto-attached keyed by their SQL alias). Display needs a small addition
# to the candidate_cards macro in _macros.html, e.g. badges linking to the
# ZTF name on Fritz / the LSST name on its broker page.
WTPX_EXTRA = [('ztfnames', 'xm.ztfname'), ('lsstnames', 'xm.lsstname')]

# Peak-to-peak / amplitude-from-nondetection measurements in source_fp_stats,
# split into diff- and stack-image groups for the 'Min. diff mag' / 'Min.
# stack mag' filters -- each group true if ANY of its 4 columns exceeds its
# own form threshold, and the two groups are AND'd together. NaN is a real
# stored float value in this table (not SQL NULL), so it must be excluded
# explicitly: Postgres treats NaN as greater than every other float, so an
# unguarded '> %s' would let NaN rows leak in.
WTPX_DIFF_AMP_COLS = [
    'diff_w1_psfmag_ptp', 'diff_w2_psfmag_ptp',
    'diff_w1_amplitude_from_nondet', 'diff_w2_amplitude_from_nondet',
]
WTPX_STACK_AMP_COLS = [
    'stack_w1_psfmag_ptp', 'stack_w2_psfmag_ptp',
    'stack_w1_amplitude_from_nondet', 'stack_w2_amplitude_from_nondet',
]

def wtp_xmatch(form):
    # Standalone builder (like wtp_yso) rather than _wtp_base -- this tab has
    # no Age field, so the hostless-tab criteria are inlined here minus the
    # mjd-firstdet age window. Column refs are qualified with cand. because
    # this tab inner-joins source_fp_stats, which has columns of the same
    # names (unqualified refs are ambiguous there).
    brd, brm = _wtp_f(form, 'brdistlim'), _wtp_f(form, 'brmaglim')
    sql = ['abs(f.gallat) >= %s AND abs(f.gallat) < %s '
        'AND cand.epochid = %s AND cand.rbscore >= %s AND cand.rbscore <= %s '
        'AND cand.nmatches >= %s AND cand.scorr_peak >= %s AND cand.ispos = 1 '
        'AND cand.distnearbrstar > %s '
        'AND (cand.wdist1 > %s OR cand.w1mag1 > %s) AND (cand.wdist2 > %s OR cand.w1mag2 > %s) '
        'AND (cand.wdist3 > %s OR cand.w1mag3 > %s)']
    p = [_wtp_f(form, 'gallimlow'), _wtp_f(form, 'gallimhigh'), _wtp_i(form, 'scanep'),
        _wtp_f(form, 'rbscorelow'), _wtp_f(form, 'rbscorehigh'), _wtp_i(form, 'nmatches'),
        _wtp_f(form, 'scorrpeak'), brd,
        brd, brm, brd, brm, brd, brm]
    # Crossmatch requirement from the form: either | ztf | lsst | both.
    req = str(form.get('xmreq', 'either')).strip().lower()
    sql.append({
        'ztf': 'xm.ztfname IS NOT NULL',
        'lsst': 'xm.lsstname IS NOT NULL',
        'both': 'xm.ztfname IS NOT NULL AND xm.lsstname IS NOT NULL',
    }.get(req, '(xm.ztfname IS NOT NULL OR xm.lsstname IS NOT NULL)'))
    # Min. diff mag / Min. stack mag: true if any of the diff group's 4
    # ptp-or-amplitude measures exceeds mindiffmag (and isn't NaN), AND
    # likewise for the stack group against minstackmag.
    mindiff = _wtp_f(form, 'mindiffmag')
    minstack = _wtp_f(form, 'minstackmag')
    for cols, thresh in ((WTPX_DIFF_AMP_COLS, mindiff), (WTPX_STACK_AMP_COLS, minstack)):
        sql.append('(%s)' % ' OR '.join(
            "(xm.%s > %%s AND xm.%s != 'NaN')" % (c, c) for c in cols))
        p += [thresh] * len(cols)
    return ' AND '.join(sql), p

WTPX_DEFAULTS = {
    # hostless-tab defaults (values mirror app.py's defhostless) + xmreq
    'rbscorelow': 0.1, 'rbscorehigh': 1.0, 'nmatches': 2,
    'gallimlow': 0.0, 'gallimhigh': 10.0,
    'brdistlim': 10.0, 'brmaglim': 7.0,
    'scorrpeak': 10.0, 'scanep': 10,
    'skipnum': 0, 'xmreq': 'either', 'mindiffmag': 3.0, 'minstackmag': 1.0,
}

_F_XMATCH = [f for f in _F_BASE if f.get('lo') != 'agelowlim'] + [
    {'kind': 'num', 'label': 'Min. diff mag', 'name': 'mindiffmag', 'suffix': 'mag'},
    {'kind': 'num', 'label': 'Min. stack mag', 'name': 'minstackmag', 'suffix': 'mag'},
    _F_BR,
    {'kind': 'wide', 'label': 'Require xmatch (either/ztf/lsst/both)', 'name': 'xmreq'},
]

# --- Pseudo-survey registration ---------------------------------------------
WTPX = _dc.replace(WTP, key='wtpx', label='NEOWISE x ZTF/LSST',
    scans={
        'xmatch': {'template': 'scan.html', 'builder': wtp_xmatch,
            'join': lambda f: (WTPX_JOIN, []), 'extra': WTPX_EXTRA,
            'defaults': WTPX_DEFAULTS,
            'title': 'NEOWISE transients crossmatched to ZTF / LSST',
            'fields': _F_XMATCH},
    })
SURVEYS[WTPX.key] = WTPX

# --- Renderer: NEOWISE figure + ZTF/LSST panel ------------------------------
def _wtpx_ext_names(cur, cand):
    """{'ztfname':..., 'lsstname':...} for this candidate, or None. Keyed
    directly by candid, same join key as WTPX_JOIN."""
    cur.execute('SELECT ztfname, lsstname FROM source_fp_stats '
        'WHERE candid = %s LIMIT 1;', (cand['candid'],))
    return cur.fetchone()

def _wtpx_render_png(survey, candid):
    """Standard WTP figure (cutouts + NEOWISE LC via _build_figure) with an
    extra ZTF/LSST panel appended below, sharing the MJD axis range."""
    conn, cur = survey.open_cursor()
    try:
        cur.execute('SELECT %s.* FROM %s WHERE %s.%s = %%s LIMIT 1'
            % (survey.alias, survey.base_from, survey.alias, survey.id_col),
            (candid,))
        cand = cur.fetchone()
        if cand is None:
            raise KeyError(candid)
        lc = wtp_fetch_lightcurve(cur, cand)
        lims = wtp_fetch_limits(cur, cand)
        name = wtp_fetch_name(cur, cand)
        cut = wtp_fetch_cutouts(cur, cand)
        fp = wtp_fetch_forced(cur, cand)
        xm = _wtpx_ext_names(cur, cand)
    finally:
        survey.close_cursor(conn, cur)

    # Broker calls AFTER the cursor is returned to the pool (they can be slow).
    ext = []                                    # [(survey, name, lc), ...]
    if xm:
        for s, k in (('ztf', 'ztfname'), ('lsst', 'lsstname')):
            if xm.get(k):
                elc = babamul_get_lightcurve(s, xm[k])
                if elc is not None and len(elc['mjd']):
                    ext.append((s, xm[k], elc))

    from matplotlib.gridspec import GridSpec
    fig = Figure(figsize=(10, 13 if cut is not None else 8))
    if cut is not None:
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.1, 1.1])
        for i, (img, ttl) in enumerate(zip(
                (cut['sci'], cut['ref'], cut['diff']),
                ('Science', 'Reference', 'Difference'))):
            a = fig.add_subplot(gs[0, i])
            _, med, std = sigma_clipped_stats(img)
            a.imshow(img, cmap='gray', vmin=med - std, vmax=med + 5 * std)
            a.set_title(ttl, fontsize=16)
            a.set_xticks([]); a.set_yticks([])
        ax_ir = fig.add_subplot(gs[1, :])
        ax_opt = fig.add_subplot(gs[2, :], sharex=ax_ir)
    else:
        gs = GridSpec(2, 1, figure=fig)
        ax_ir = fig.add_subplot(gs[0])
        ax_opt = fig.add_subplot(gs[1], sharex=ax_ir)

    # -- NEOWISE panel: same conventions as _build_figure (forced solid,
    # candidate-based faded, limits as arrows, stack forced as open diamonds).
    has_fp = bool(fp and len(fp['mjd']))
    cand_alpha = 0.3 if has_fp else 1.0
    fsmag = np.asarray(fp.get('smag', [])) if has_fp else np.array([])
    fsmagerr = np.asarray(fp.get('smagerr', [])) if has_fp else np.array([])
    fsdet = np.asarray(fp.get('sisdet', []), dtype=bool) if has_fp else np.array([], dtype=bool)
    for b in survey.bands:
        d = (lc['bandid'] == b.id)
        if np.any(d):
            ax_ir.errorbar(lc['mjd'][d], lc['mag'][d],
                yerr=np.abs(lc['magerr'][d]), ls='none', marker='s',
                color=b.color, ms=9, alpha=cand_alpha,
                label=(None if has_fp else b.label))
        if has_fp:
            fb = (np.asarray(fp['bandid']) == b.id)
            det = fb & np.asarray(fp['isdet'], dtype=bool)
            lim = fb & ~np.asarray(fp['isdet'], dtype=bool)
            if np.any(det):
                ax_ir.errorbar(fp['mjd'][det], fp['mag'][det],
                    yerr=np.abs(fp['magerr'][det]), ls='none', marker='o',
                    color=b.color, ms=10, label=b.label)
            if np.any(lim):
                ax_ir.errorbar(fp['mjd'][lim], fp['limmag'][lim], yerr=0.2,
                    uplims=True, ls='none', marker='v', color=b.color,
                    markerfacecolor='none', ms=8)
            # forced STACK photometry -- open diamonds + thin line, no limits
            # (same convention as _build_figure).
            if len(fsmag):
                sb = fb & fsdet & np.isfinite(fsmag)
                if np.any(sb):
                    ax_ir.errorbar(fp['mjd'][sb], fsmag[sb], yerr=np.abs(fsmagerr[sb]),
                        ls='-', lw=0.8, marker='D', mfc='none', color=b.color, ms=8)
    ax_ir.set_ylabel('NEOWISE mag', fontsize=16)
    ax_ir.set_title('Candidate %d; %s' % (candid, name), fontsize=16)
    ax_ir.invert_yaxis()
    handles, labels = ax_ir.get_legend_handles_labels()
    if handles:
        if has_fp and len(fsmag) and fsdet.any():
            from matplotlib.lines import Line2D
            handles = handles + [
                Line2D([], [], color='0.3', marker='o', ls='none', ms=9),
                Line2D([], [], color='0.3', marker='D', mfc='none', ls='none', ms=8)]
            labels = labels + ['diff forced', 'stack forced']
        ax_ir.legend(handles, labels, fontsize=10)

    # -- ZTF/LSST panel from Babamul --
    for s, nm, elc in ext:
        mk = EXT_MARKERS[s]
        isneg = elc.get('isneg', np.zeros(len(elc['mjd']), dtype=bool))
        for f in np.unique(elc['filt']):
            sel = (elc['filt'] == f)
            col = EXT_BAND_COLORS.get((s, str(f)), 'k')
            det = sel & elc['isdet']
            lim = sel & ~elc['isdet'] & np.isfinite(elc['limmag'])
            det_pos = det & ~isneg
            det_neg = det & isneg
            if np.any(det_pos):
                ax_opt.errorbar(elc['mjd'][det_pos], elc['mag'][det_pos],
                    yerr=np.abs(elc['magerr'][det_pos]), ls='none', marker=mk,
                    color=col, ms=7, label='%s %s' % (s.upper(), f))
            if np.any(det_neg):
                # isdiffpos == False -- negative-flux ('negative') detection,
                # marked with a cross instead of the normal survey marker.
                ax_opt.errorbar(elc['mjd'][det_neg], elc['mag'][det_neg],
                    yerr=np.abs(elc['magerr'][det_neg]), ls='none', marker='x',
                    color=col, ms=7,
                    label=(None if np.any(det_pos) else '%s %s (neg)' % (s.upper(), f)))
            if np.any(lim):
                ax_opt.errorbar(elc['mjd'][lim], elc['limmag'][lim], yerr=0.2,
                    uplims=True, ls='none', marker='v', color=col,
                    markerfacecolor='none', ms=6, alpha=0.5)
    ax_opt.set_ylabel('ZTF / LSST mag', fontsize=16)
    ax_opt.set_xlabel('MJD', fontsize=16)
    ax_opt.invert_yaxis()
    if ext:
        ax_opt.legend(fontsize=9, ncol=3)
        ax_opt.set_title(' / '.join('%s' % nm for _, nm, _e in ext), fontsize=11)
    else:
        ax_opt.text(0.5, 0.5, 'no broker photometry available',
            transform=ax_opt.transAxes, ha='center', color='0.5')

    # Candidate-epoch marker on both panels (blue dashed, as elsewhere).
    _cmjd = cand.get('mjd')
    if _cmjd is not None and np.isfinite(float(_cmjd)):
        for axx in (ax_ir, ax_opt):
            axx.axvline(float(_cmjd), color='b', ls='--', lw=1.0, alpha=0.7)
    if len(lc['mjd']) or (ext and len(ext[0][2]['mjd'])):
        ax2 = ax_ir.twiny()
        ax2.set_xlim(Time(ax_ir.get_xlim(), format='mjd').decimalyear)
        ax2.set_xlabel('Year', fontsize=14)

    fig.tight_layout()
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80)
    return buf.getvalue()

# Dispatch: wtpx goes through the crossmatch renderer; everything else keeps
# its existing path (chains behind the PRIME wrapper above).
_png_pre_wtpx = candidate_png_by_id
def candidate_png_by_id(survey, candid):
    if getattr(survey, 'key', None) == 'wtpx':
        return _wtpx_render_png(survey, candid)
    return _png_pre_wtpx(survey, candid)
# <<< NEOWISE x ZTF/LSST crossmatch scan


# >>> Custom SQL scan -----------------------------------------------------
# ---------------------------------------------------------------------------
# A tab where the user writes an arbitrary SELECT statement, sees how many
# candidates it resolves to, and -- only if they confirm -- gets the normal
# scan.html results page (cards, cutouts, pagination) for those candidates.
#
# Two-step flow:
#   1. wtp_run_customsql(sql_text) executes the user's SELECT in a Postgres
#      READ ONLY transaction with a statement timeout, wrapped so only a
#      `candid` output column is ever pulled out. The READ ONLY transaction
#      is a real DB-level guarantee (verified: Postgres raises
#      ReadOnlySqlTransaction on INSERT/UPDATE/DELETE/DDL regardless of
#      wording) -- it is the actual security boundary here, not the
#      single-statement check below, which is just a usability guard.
#      The resolved candid list is cached server-side under a random token
#      (app.py never re-executes the user's SQL on confirm).
#   2. The 'customsql_run' WTP scan's builder looks up that token and
#      returns 'cand.candid = ANY(%s)' -- from there it's just another scan,
#      reusing run_candidate_query/serialize_candidates/scan.html exactly as
#      every other tab does (dedup by name, sort, candlim/skipnum paging,
#      on-demand cutout PNGs via the existing /wtp/cutout/<candid>.png route).
# ---------------------------------------------------------------------------
import secrets as _secrets

CUSTOMSQL_TTL = float(os.environ.get('CUSTOMSQL_TTL', '600'))          # 10 min
CUSTOMSQL_ROW_CAP = int(os.environ.get('CUSTOMSQL_ROW_CAP', '5000'))
CUSTOMSQL_TIMEOUT_MS = int(os.environ.get('CUSTOMSQL_TIMEOUT_MS', '30000'))
_CUSTOMSQL_CACHE = {}
_CUSTOMSQL_LOCK = threading.Lock()

def _customsql_gc():
    now = time.time()
    with _CUSTOMSQL_LOCK:
        for k in [k for k, v in _CUSTOMSQL_CACHE.items()
                if now - v['ts'] > CUSTOMSQL_TTL]:
            del _CUSTOMSQL_CACHE[k]

def customsql_store(sql_text, candids):
    _customsql_gc()
    token = _secrets.token_urlsafe(16)
    with _CUSTOMSQL_LOCK:
        _CUSTOMSQL_CACHE[token] = {'ts': time.time(), 'sql': sql_text,
            'candids': candids}
    return token

def customsql_lookup(token):
    with _CUSTOMSQL_LOCK:
        return _CUSTOMSQL_CACHE.get(token)

def wtp_run_customsql(sql_text):
    """Execute a user-supplied read-only SQL query and resolve it to a list
    of candids. Returns (candids, total, truncated).

    Raises ValueError (safe to show to the user -- they wrote the query) on
    an empty/multi-statement query, a bad column name, a rejected mutation,
    a timeout, or any other Postgres error.
    """
    text = sql_text.strip()
    if not text:
        raise ValueError('Query is empty.')
    if text.endswith(';'):
        text = text[:-1].rstrip()
    if ';' in text:
        raise ValueError("Only a single SQL statement is allowed "
            "(remove the extra ';').")
    wrapped = ('SELECT DISTINCT candid FROM (%s) AS _customsql_user_query '
        'LIMIT %%s' % text)
    conn, cur = _wtp_open()
    try:
        conn.set_session(readonly=True)
        cur.execute('SET statement_timeout = %s', (CUSTOMSQL_TIMEOUT_MS,))
        cur.execute(wrapped, (CUSTOMSQL_ROW_CAP + 1,))
        rows = cur.fetchall()
    except psycopg2.Error as e:
        raise ValueError(str(e).strip())
    finally:
        try:
            conn.set_session(readonly=False)
        except Exception:
            pass
        _wtp_close(conn, cur)
    truncated = len(rows) > CUSTOMSQL_ROW_CAP
    candids = [r['candid'] for r in rows[:CUSTOMSQL_ROW_CAP]]
    return candids, len(candids), truncated

def wtp_customsql_where(form):
    token = form.get('customsql_token', '')
    entry = customsql_lookup(token)
    if entry is None:
        raise KeyError('Custom-SQL result expired or not found; '
            'go back and resubmit the query.')
    return 'cand.candid = ANY(%s)', [entry['candids']]

WTP.scans['customsql_run'] = {
    'template': 'scan.html', 'builder': wtp_customsql_where,
    'title': 'Custom SQL results', 'fields': [],
}
# <<< Custom SQL scan
