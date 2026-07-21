"""
WTP scanning web app — complete, self-contained.
Requires: surveys.py alongside this file, templates/ and static/ populated.
"""
import os
import io
import gzip
import json
import socket
import base64
import getpass
import logging
import threading
from io import BytesIO
from datetime import datetime, timedelta, UTC
from functools import wraps, lru_cache

import numpy as np
import requests
from astropy.io import fits, ascii
from astropy.stats import sigma_clipped_stats
from flask import (Flask, render_template, request, session, flash,
    abort, jsonify, Response, redirect)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix

import surveys
from surveys import (SURVEYS, build_scan_context, serialize_candidates,
    candidate_png_by_id, fritz_mark_saved, ascii_name_list)

logger = logging.getLogger(__name__)
WTP = SURVEYS['wtp']
s_cm_radius = 5.0 / 3600
_WTP_COLS = ', '.join('cand.%s' % c for _, c in WTP.field_map)

# ---------------------------------------------------------------------------
# Config (secrets from env only)
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get('WTP_SECRET_KEY')
if not SECRET_KEY:
    raise RuntimeError('WTP_SECRET_KEY is not set. Generate one with '
        '`python -c "import secrets; print(secrets.token_hex(32))"`.')
USERS_FILE = os.environ.get('WTP_USERS_FILE', 'users.json')
FRITZ_TOKEN = os.environ.get('FRITZ_TOKEN', '')
ENFORCE_CSRF = os.environ.get('WTP_CSRF', '0') == '1'

BASEURL = 'https://fritz.science/'
neowise_fritz_id = 408
neowise_inst_id = 73

# ---------------------------------------------------------------------------
# Default parameter dicts
# ---------------------------------------------------------------------------
defrblow, defrbhigh = 0.0, 1.0
defnmatches = 2
defgpl, defgph = 0.0, 10.0
defngpl, defngph = 10.0, 90.0
defbrd, defbrm = 10.0, 7.0
defagel, defageh = 10.0, 400.0
defskip = 0
defhlw = 3.0
defhls, defhle = 10.0, 10
defclud, defclus, defclue = 120.0, 5.0, 10
defhwd, defhwa, defhs, defhe = 3.0, 3.0, 10.0, 10
deflmcd = defsmcd = 300 * 60 / 3600
defm31d = 240 * 60 / 3600

defhostless = {'rbscorelow': defrblow, 'rbscorehigh': defrbhigh, 'nmatches': defnmatches, 'gallimlow': defgpl, 'gallimhigh': defgph, 'hlwdist': defhlw, 'brdistlim': defbrd, 'brmaglim': defbrm, 'agelowlim': defagel, 'agehighlim': defageh, 'scorrpeak': defhls, 'scanep': defhle, 'skipnum': defskip}
defhosted = {'rbscorelow': defrblow, 'rbscorehigh': defrbhigh, 'nmatches': defnmatches, 'gallimlow': defgpl, 'gallimhigh': defgph, 'hwdist': defhwd, 'hwamp': defhwa, 'brdistlim': defbrd, 'brmaglim': defbrm, 'agelowlim': defagel, 'agehighlim': defageh, 'scorrpeak': defhs, 'scanep': defhe, 'skipnum': defskip}
defnuclear = {'rbscorelow': defrblow, 'rbscorehigh': defrbhigh, 'nmatches': defnmatches, 'gallimlow': defngpl, 'gallimhigh': defngph, 'cludist': 2.0, 'brdistlim': defbrd, 'brmaglim': defbrm, 'agelowlim': defagel, 'agehighlim': defageh, 'scorrpeak': defclus, 'scanep': defclue, 'skipnum': defskip}
defyso = {'rbscorelow': defrblow, 'rbscorehigh': defrbhigh, 'gallimlow': 0, 'gallimhigh': 5, 'declimlow': -90, 'declimhigh': 90, 'ralimlow': 0, 'ralimhigh': 360, 'scorrpeak': 10.0, 'durmag': 13.0, 'minduration': 3.0, 'colorlimlow': 0.5, 'colorlimhigh': 3.5, 'curmaglow': 0.0, 'curmaghigh': 13.0, 'brdistlim': defbrd, 'brmaglim': defbrm, 'scanep': 18, 'skipnum': defskip}
deflmc = {'rbscorelow': defrblow, 'rbscorehigh': defrbhigh, 'nmatches': defnmatches, 'gallimlow': defngpl, 'gallimhigh': defngph, 'lmcdist': deflmcd, 'hlwdist': defhlw, 'hwdist': defhwd, 'hwamp': defhwa, 'brdistlim': defbrd, 'brmaglim': defbrm, 'agelowlim': defagel, 'agehighlim': defageh, 'scorrpeak': defclus, 'scanep': defclue, 'skipnum': defskip}
defsmc = {'rbscorelow': defrblow, 'rbscorehigh': defrbhigh, 'nmatches': defnmatches, 'gallimlow': defngpl, 'gallimhigh': defngph, 'smcdist': defsmcd, 'hlwdist': defhlw, 'hwdist': defhwd, 'hwamp': defhwa, 'brdistlim': defbrd, 'brmaglim': defbrm, 'agelowlim': defagel, 'agehighlim': defageh, 'scorrpeak': defclus, 'scanep': defclue, 'skipnum': defskip}
defm31 = {'rbscorelow': defrblow, 'rbscorehigh': defrbhigh, 'nmatches': defnmatches, 'gallimlow': defngpl, 'gallimhigh': defngph, 'm31dist': defm31d, 'hlwdist': 2.0, 'hwdist': 2.0, 'hwamp': 1.0, 'brdistlim': 6.0, 'brmaglim': defbrm, 'agelowlim': defagel, 'agehighlim': defageh, 'scorrpeak': defclus, 'scanep': defclue, 'skipnum': defskip}
defclu = {'rbscorelow': defrblow, 'rbscorehigh': defrbhigh, 'nmatches': defnmatches, 'gallimlow': defngpl, 'gallimhigh': defngph, 'cludist': defclud, 'brdistlim': defbrd, 'brmaglim': defbrm, 'agelowlim': defagel, 'agehighlim': defageh, 'scorrpeak': defclus, 'scanep': defclue, 'skipnum': defskip}
defsearch = {'rasearch': 150.00, 'decsearch': 30.00, 'cmradius': 10.0, 'wname': 'WTP16aaabvl'}

# Map scan names to their default parameter dicts
SCAN_DEFAULTS = {
    'hostless': defhostless, 'hosted': defhosted, 'clu': defclu,
    'nuclear': defnuclear, 'smc': defsmc, 'm31': defm31, 'lmc': deflmc,
    'yso': defyso,
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')

if ENFORCE_CSRF:
    from flask_wtf import CSRFProtect
    from flask_wtf.csrf import CSRFError
    CSRFProtect(app)
    app.config['WTF_CSRF_TIME_LIMIT'] = None    # token lives as long as the session

    @app.errorhandler(CSRFError)
    def _err_csrf(e):
        return ('Form token expired or invalid. Reload the page and submit '
            'again.'), 400

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    SESSION_COOKIE_SECURE=(os.environ.get('WTP_HTTPS', '0') == '1'),
)

logging.basicConfig(level=os.environ.get('WTP_LOG_LEVEL', 'INFO'),
    format='%(asctime)s %(levelname)s %(name)s %(message)s')

@app.after_request
def _security_headers(resp):
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('X-Frame-Options', 'DENY')
    resp.headers.setdefault('Referrer-Policy', 'same-origin')
    return resp

@app.errorhandler(400)
def _err_400(e):
    return 'Bad request: check your scan parameters.', 400

@app.errorhandler(404)
def _err_404(e):
    return 'Not found.', 404

@app.errorhandler(500)
def _err_500(e):
    app.logger.exception('Unhandled server error')
    return 'Internal server error. The issue has been logged.', 500

if os.environ.get('WTP_RATELIMIT', '0') == '1':
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app,
        storage_uri=os.environ.get('WTP_RATELIMIT_STORAGE', 'memory://'))
    login_limit = limiter.limit(os.environ.get('WTP_LOGIN_RATELIMIT', '10 per minute'))
else:
    def login_limit(fn):
        return fn

def now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def load_users():
    try:
        with open(USERS_FILE) as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.error('Users file %s not found', USERS_FILE)
        return {}

def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **k):
        if not session.get('logged_in'):
            return render_template('login.html')
        return fn(*a, **k)
    return wrapper

@app.route('/login', methods=['POST'])
@login_limit
def do_user_login():
    users = load_users()
    u, p = request.form.get('username', ''), request.form.get('password', '')
    stored = users.get(u)
    if stored and check_password_hash(stored, p):
        session.clear()
        session['logged_in'] = True
        session['user'] = u
        return redirect('/')
    check_password_hash(generate_password_hash('x'), p)
    flash('Invalid username or password.')
    return main()

@app.route('/logout')
def logout():
    session.clear()
    return render_template('logout.html')

# ---------------------------------------------------------------------------
# Cutout render route (on-demand, cached in-process)
# ---------------------------------------------------------------------------
# Auto-derived from surveys.py's mtime: any edit to the figure-rendering code
# changes the ETag on the next app restart, so client caches bust automatically
# -- no manual version bumping, no stale PNGs.
import surveys as _surveys_mod
CUTOUT_RENDER_VERSION = str(int(os.path.getmtime(_surveys_mod.__file__)))

def _render_day():
    """UTC day bucket folded into render cache keys and ETags so figures
    pick up the nightly forced-photometry refresh without a restart."""
    return datetime.now(UTC).strftime('%Y%m%d')

@lru_cache(maxsize=2048)
def _cached_cutout(survey_key, candid, day):
    return surveys.candidate_png_by_id(SURVEYS[survey_key], candid)

@app.route('/<survey_key>/cutout/<int:candid>.png')
@login_required
def cutout(survey_key, candid):
    if survey_key not in SURVEYS:
        abort(404)
    # ETag carries a render-version tag so changing the figure code invalidates
    # client caches; max-age=0 + must-revalidate forces a conditional GET each
    # view (cheap: the PNG is memoized in-process by _cached_cutout).
    day = _render_day()
    etag = '"%s-%d-%s-%s"' % (survey_key, candid, CUTOUT_RENDER_VERSION, day)
    headers = {'Cache-Control': 'private, max-age=0, must-revalidate', 'ETag': etag}
    if request.headers.get('If-None-Match') == etag:
        return Response(status=304, headers=headers)
    try:
        png = _cached_cutout(survey_key, candid, day)
    except KeyError:
        abort(404)
    return Response(png, mimetype='image/png', headers=headers)

# ---------------------------------------------------------------------------
# Generic scan route (new-style: GET = form, POST = results)
# ---------------------------------------------------------------------------
@app.route('/<survey_key>/scan/<scan_name>', methods=['GET', 'POST'])
@login_required
def run_scan(survey_key, scan_name):
    survey = SURVEYS.get(survey_key)
    if survey is None or scan_name not in survey.scans:
        abort(404)
    spec = survey.scans[scan_name]
    if request.method == 'GET':
        defaults = dict(spec.get('defaults', SCAN_DEFAULTS.get(scan_name, {})))
        if 'datemax' in defaults:
            # Prefill the datetime-local window with exactly last night in UTC:
            # [now - 1 day, now]. now is sampled once so the two bounds are
            # exactly 24 h apart (rendering at MJD 61242.5 -> the builder
            # resolves the SQL to [61241.5, 61242.5)). datetime-local is
            # minute-precision, which keeps the span exact; the builder does the
            # MJD conversion. Typing narrower/wider bounds overrides this.
            # [patch prime-datetime-window-20260721]
            _now = datetime.now(UTC)
            defaults['datemax'] = _now.strftime('%Y-%m-%dT%H:%M')
            if 'datemin' in defaults:
                defaults['datemin'] = (_now - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M')
        return render_template('scan.html', user=session['user'],
            server=socket.gethostname(), timenow=now_str(),
            scan_title=spec.get('title', scan_name),
            scan_fields=spec.get('fields', []),
            scan_action='/%s/scan/%s' % (survey_key, scan_name),
            defpardict=defaults,
            numcands=0, canddicts=None)
    try:
        template, ctx = surveys.build_scan_context(survey, scan_name, request.form)
    except (KeyError, ValueError, TypeError):
        app.logger.exception('build_scan_context failed (generic route)')
        abort(400)
    return render_template(template, user=session['user'],
        server=socket.gethostname(), timenow=now_str(), **ctx)

# ---------------------------------------------------------------------------
# Custom SQL scan: user writes a SELECT, sees a candidate count, and only on
# confirm gets the normal scan.html results page. Query execution itself
# (read-only enforcement, wrapping, caching the resolved candids under a
# token) lives in surveys.wtp_run_customsql -- this view is just the
# count-then-confirm form around it. The confirm step is the existing
# generic /wtp/scan/customsql_run route (registered in surveys.py), not
# handled here.
# ---------------------------------------------------------------------------
@app.route('/wtp/customsql', methods=['GET', 'POST'])
@login_required
def wtp_customsql():
    sql_text = request.form.get('sql', '')
    error = None
    result = None
    if request.method == 'POST':
        try:
            candids, total, truncated = surveys.wtp_run_customsql(sql_text)
        except ValueError as e:
            error = str(e)
        else:
            token = surveys.customsql_store(sql_text, candids)
            result = {'total': total, 'truncated': truncated, 'token': token}
    return render_template('customsql.html', user=session['user'],
        server=socket.gethostname(), timenow=now_str(),
        sql_text=sql_text, error=error, result=result,
        row_cap=surveys.CUSTOMSQL_ROW_CAP)

# ---------------------------------------------------------------------------
# Legacy scan POST routes (/wise_*_scan)
# ---------------------------------------------------------------------------
def _make_legacy_scan(name):
    @login_required
    def view():
        try:
            template, ctx = surveys.build_scan_context(WTP, name, request.form)
        except (KeyError, ValueError, TypeError):
            app.logger.exception('build_scan_context failed (legacy route)')
            abort(400)
        return render_template(template, user=session['user'],
            server=socket.gethostname(), timenow=now_str(), **ctx)
    return view

for _name in SCAN_DEFAULTS:
    app.add_url_rule('/wise_%s_scan' % _name, 'legacy_scan_' + _name,
        _make_legacy_scan(_name), methods=['POST'])

# ---------------------------------------------------------------------------
# Legacy scan index GET routes (/wise_*)
# ---------------------------------------------------------------------------
def _make_scan_index(scan_name, defaults):
    """Return a view function that renders the scan form (no results)."""
    @login_required
    def view():
        spec = WTP.scans[scan_name]
        return render_template('scan.html', user=session['user'],
            server=socket.gethostname(), timenow=now_str(),
            scan_title=spec.get('title', scan_name),
            scan_fields=spec.get('fields', []),
            scan_action='/wise_%s_scan' % scan_name,
            defpardict=defaults, numcands=0, canddicts=None)
    return view

for _name, _defaults in SCAN_DEFAULTS.items():
    app.add_url_rule('/wise_%s' % _name, 'idx_' + _name,
        _make_scan_index(_name, _defaults))

# ---------------------------------------------------------------------------
# Name / spatial search
# ---------------------------------------------------------------------------
@app.route('/wise_search_name', methods=['POST'])
@login_required
def name_search():
    conn, cur = WTP.open_cursor()
    try:
        sql = ('SELECT %s FROM candidates cand '
            'JOIN sourcenames s ON q3c_join(s.ra, s.dec, cand.ra, cand.dec, %%s) '
            'WHERE s.name = %%s ORDER BY cand.scorr_peak DESC LIMIT 1;' % _WTP_COLS)
        qparams = (s_cm_radius, request.form['wname'])
        qstr = cur.mogrify(sql, qparams).decode()
        app.logger.info(qstr)
        cur.execute(sql, qparams)
        out = cur.fetchall()
    finally:
        WTP.close_cursor(conn, cur)
    cd = serialize_candidates(WTP, out)
    return render_template('wise_show_cands.html', user=session['user'],
        server=socket.gethostname(), numcands=len(cd['candids']),
        timenow=now_str(), canddicts=cd, scan_query=qstr,
        ascii_list=ascii_name_list(cd),
        defpardict=request.form.to_dict())

@app.route('/wise_search_spatial', methods=['POST'])
@login_required
def spatial_search(defcandlim=200):
    conn, cur = WTP.open_cursor()
    try:
        sql = ('SELECT %s FROM candidates cand '
            'WHERE q3c_radial_query(cand.ra, cand.dec, %%s, %%s, %%s) '
            'ORDER BY cand.scorr_peak DESC LIMIT %%s;' % _WTP_COLS)
        qparams = (float(request.form['rasearch']), float(request.form['decsearch']),
            float(request.form['cmradius']) / 3600, defcandlim)
        qstr = cur.mogrify(sql, qparams).decode()
        app.logger.info(qstr)
        cur.execute(sql, qparams)
        out = cur.fetchall()
    finally:
        WTP.close_cursor(conn, cur)
    cd = serialize_candidates(WTP, out)
    return render_template('wise_show_cands.html', user=session['user'],
        server=socket.gethostname(), numcands=len(cd['candids']),
        timenow=now_str(), canddicts=cd, scan_query=qstr,
        ascii_list=ascii_name_list(cd),
        defpardict=request.form.to_dict())

# ---------------------------------------------------------------------------
# Fritz
# ---------------------------------------------------------------------------
def fritz_api(method, endpoint, data=None):
    return requests.request(method, endpoint, json=data,
        headers={'Authorization': 'token %s' % FRITZ_TOKEN}, timeout=30).json()

# W1/W2 proxies registered on the Fritz instrument (JWST bandpass names
# standing in for WISE, as in the original pipeline).
FRITZ_FILTERS = {1: 'f356w', 2: 'f444w'}

def _fritz_put_phot(sourcename, mjd, mag, magerr, limmag, band,
        limflag, nsigma=5.0):
    payload = {'instrument_id': str(neowise_inst_id), 'obj_id': sourcename,
        'mjd': [float(x) for x in mjd],
        'limiting_mag': [float(x) for x in limmag],
        'limiting_mag_nsigma': nsigma, 'magsys': 'vega',
        'filter': [FRITZ_FILTERS.get(int(b), 'f356w') for b in band],
        'group_ids': [neowise_fritz_id]}
    if not limflag:
        payload['mag'] = [float(x) for x in mag]
        payload['magerr'] = [abs(float(x)) for x in magerr]
    resp = fritz_api('PUT', BASEURL + 'api/photometry', data=payload)
    if resp.get('status') != 'success':
        app.logger.error('Fritz photometry upload failed for %s: %s',
            sourcename, resp)

def fritz_upload_photometry(sourcename, cand):
    """Port of post_fritz_photo: detections with per-epoch limiting mags,
    then non-detection upper limits from the candidate's dominant field."""
    conn, cur = WTP.open_cursor()
    try:
        cur.execute('SELECT cand.mjd, cand.psf_mag, cand.psf_mag_err, '
            'cand.bandid, cand.field, sub.limmag FROM candidates cand '
            'INNER JOIN subtractions sub ON cand.subid = sub.subid '
            'WHERE q3c_radial_query(ra, dec, %s, %s, %s) AND ispos = 1 '
            'ORDER BY mjd DESC;', (cand['ra'], cand['dec'], s_cm_radius))
        out = cur.fetchall()
        if not out:
            return
        fields = np.array([o['field'] for o in out])
        unf, cnt = np.unique(fields, return_counts=True)
        fielduse = int(unf[np.argmax(cnt)])
        mjd = np.array([o['mjd'] for o in out])
        mag = np.array([o['psf_mag'] for o in out])
        magerr = np.array([o['psf_mag_err'] for o in out])
        band = np.array([o['bandid'] for o in out])
        limmag = np.array([o['limmag'] for o in out])
        # drop duplicate epochs contributed by overlapping fields
        _, idx = np.unique(np.round(1e6 * band + mjd + mag, 5),
            return_index=True)
        _fritz_put_phot(sourcename, mjd[idx], mag[idx], magerr[idx],
            limmag[idx], band[idx], False)
        # non-detections: epochs of the dominant field with no counterpart
        cur.execute('SELECT st.mjdmin AS mjd, sub.bandid AS bandid, '
            'sub.limmag AS limmag FROM subtractions sub '
            'INNER JOIN stacks st ON st.stackid = sub.stackid '
            'WHERE sub.field = %s AND numposcand >= 0 AND subid NOT IN '
            '(SELECT subid FROM candidates WHERE '
            'q3c_radial_query(ra, dec, %s, %s, %s) AND ispos = 1);',
            (fielduse, cand['ra'], cand['dec'], s_cm_radius))
        lo = cur.fetchall()
    finally:
        WTP.close_cursor(conn, cur)
    if lo:
        _fritz_put_phot(sourcename,
            np.array([o['mjd'] for o in lo]), None, None,
            np.array([o['limmag'] for o in lo]),
            np.array([o['bandid'] for o in lo]), True)

def fritz_upload_thumbnails(sourcename, cand, lowp=1, highp=5):
    """Port of post_fritz_cutouts: sci/ref/diff as Fritz new/ref/sub."""
    if WTP.fetch_cutouts is None:
        return
    conn, cur = WTP.open_cursor()
    try:
        cut = WTP.fetch_cutouts(cur, cand)
    finally:
        WTP.close_cursor(conn, cur)
    if cut is None:
        return
    from matplotlib import image as mpimg
    for ttype, key in (('new', 'sci'), ('ref', 'ref'), ('sub', 'diff')):
        img = cut[key]
        _, med, std = sigma_clipped_stats(img)
        buf = BytesIO()
        mpimg.imsave(buf, img, cmap='gray', vmin=med - lowp * std,
            vmax=med + highp * std, format='png')
        resp = fritz_api('POST', BASEURL + 'api/thumbnail',
            data={'obj_id': sourcename, 'ttype': ttype,
            'data': base64.b64encode(buf.getvalue()).decode()})
        if resp.get('status') != 'success':
            app.logger.error('Fritz %s thumbnail failed for %s: %s',
                ttype, sourcename, resp)

@app.route('/wise_fritz', methods=['POST'])
@login_required
def check_fritz():
    sourcename = request.form['sourcename']
    status = fritz_api('GET', BASEURL + 'api/sources/%s' % sourcename)
    if status.get('status') == 'success':
        return render_template('get_fritz.html', savestatus=True, sourcename=sourcename)
    # Not on Fritz yet: save it to the NEOWISE group, then seed the saved-
    # names cache so the fr_ex badge picks it up on the next scan render.
    payload = {'id': sourcename, 'ra': float(request.form['ra']),
        'dec': float(request.form['dec']), 'group_ids': [neowise_fritz_id],
        'varstar': False, 'is_roid': False, 'transient': True,
        'origin': 'mwtp'}
    resp = fritz_api('POST', BASEURL + 'api/sources', data=payload)
    if resp.get('status') == 'success':
        fritz_mark_saved(sourcename)
        cand = {'candid': int(request.form['candid']),
            'ra': float(request.form['ra']), 'dec': float(request.form['dec'])}
        for fn in (fritz_upload_thumbnails, fritz_upload_photometry):
            try:
                fn(sourcename, cand)
            except Exception:
                app.logger.exception('%s failed for %s', fn.__name__, sourcename)
    else:
        app.logger.error('Fritz save failed for %s: %s', sourcename, resp)
    return render_template('get_fritz.html', savestatus=False, sourcename=sourcename)

# ---------------------------------------------------------------------------
# PRIME -> TOM save
# ---------------------------------------------------------------------------
@app.route('/prime/tom_save', methods=['POST'])
@login_required
def prime_tom_save():
    import threading
    import surveys as _s
    group = request.form.get('tomgroup', '')
    if group not in _s.PRIME_TOM_LISTS:
        return 'TOM list %r not configured (PRIME_TOM_LISTS)' % group, 400
    name = request.form['sourcename']
    conn, cur = _s._prime_open()
    try:
        cur.execute('SELECT primeid FROM sources WHERE name = %s LIMIT 1;', (name,))
        r = cur.fetchone()
    finally:
        _s._prime_close(conn, cur)
    if r is None:
        return 'Unknown PRIME source %s' % name, 404
    try:
        tid, created = _s.prime_tom_get_or_create_target(
            name, float(request.form['ra']), float(request.form['dec']), group)
    except Exception as e:
        app.logger.exception('TOM save failed for %s', name)
        return 'TOM save failed: %s' % e, 502
    threading.Thread(target=_s.prime_tom_upload_photometry,
                     args=(tid, int(r['primeid'])), daemon=True).start()
    return render_template('get_tom.html', sourcename=name, created=created,
                           group=_s.PRIME_TOM_LISTS[group],
                           tom_url=_s.TOM_BASE_URL, target_id=tid)


# ---------------------------------------------------------------------------
# PRIME -> Slack alert  [prime_slack_alert]  (patch prime-slack-alert-20260719)
# Renders the scan-card figure for a candid and posts it to Slack with a
# caption (alerter, name, mag/filter, coords, field, fpapos). Config in env:
# SLACK_ALERT_TOKEN|SLACK_BOT_TOKEN, SLACK_ALERT_CHANNEL|SLACK_CHANNEL.
# ---------------------------------------------------------------------------
@app.route('/prime/slack_alert', methods=['POST'])
@login_required
def prime_slack_alert():
    import surveys as _s
    try:
        candid = int(request.form['candid'])
    except (KeyError, ValueError):
        return 'Missing/invalid candid', 400
    alerter = session.get('user', 'unknown')
    try:
        name, permalink = _s.prime_post_slack_alert(SURVEYS['prime'], candid, alerter)
    except KeyError:
        abort(404)
    except Exception as e:
        app.logger.exception('Slack alert failed for candid=%s', candid)
        return 'Slack alert failed: %s' % e, 502
    link = ('<p><a href="%s" target="_blank" rel="noopener">View in Slack</a></p>'
            % permalink) if permalink else ''
    return ('<!doctype html><meta charset="utf-8"><title>Slack alert</title>'
            '<div style="font-family:sans-serif;padding:2em">'
            '<h2>Posted PRIME alert for %s</h2>'
            '<p>candid %d posted by %s.</p>%s'
            '<p><a href="javascript:history.back()">&larr; back</a></p></div>'
            % (name, candid, alerter, link))


# ---------------------------------------------------------------------------
# PRIME per-source viewer  (by PRIME ID or ra/dec)   [prime_source_viewer]
# ---------------------------------------------------------------------------
@app.route('/prime/source')
@login_required
def prime_source_index():
    # GET with ?sourcename= (or ?rasearch=&decsearch=) renders results
    # directly, so flagged-list names can deep-link here; bare GET shows
    # the search form.  [patch prime-flagged-20260719]
    if request.args.get('sourcename') or (
            request.args.get('rasearch') and request.args.get('decsearch')):
        try:
            template, ctx = surveys.build_prime_source_context(
                SURVEYS['prime'], request.args)
        except (KeyError, ValueError, TypeError):
            app.logger.exception('build_prime_source_context (GET) failed')
            abort(400)
        return render_template(template, user=session['user'],
            server=socket.gethostname(), timenow=now_str(), **ctx)
    return render_template('prime_source.html', user=session['user'],
        server=socket.gethostname(), timenow=now_str(),
        defpardict=surveys.PRIME_SOURCE_DEFAULTS)

@app.route('/prime/flagged')
@login_required
def prime_flagged():
    import surveys as _s
    list_name = request.args.get('list', _s.PRIME_FLAGGED_TARGETLIST)
    error, rows = None, []
    try:
        rows = _s.prime_flagged_rows(list_name)
    except Exception as e:
        app.logger.exception('flagged list fetch failed')
        error = str(e)
    return render_template('prime_flagged.html', user=session['user'],
        server=socket.gethostname(), timenow=now_str(),
        rows=rows, error=error, list_name=list_name,
        label=_s.PRIME_FLAGGED_LABEL)

@app.route('/prime/source_search', methods=['POST'])
@login_required
def prime_source_search():
    prime = SURVEYS['prime']
    try:
        template, ctx = surveys.build_prime_source_context(prime, request.form)
    except (KeyError, ValueError, TypeError):
        app.logger.exception('build_prime_source_context failed')
        abort(400)
    return render_template(template, user=session['user'],
        server=socket.gethostname(), timenow=now_str(), **ctx)

@lru_cache(maxsize=512)
def _cached_prime_source_png(primeid, day):
    return surveys.prime_render_source_png(SURVEYS['prime'], primeid)

@app.route('/prime/source_png/<int:primeid>.png')
@login_required
def prime_source_png(primeid):
    day = _render_day()
    etag = '"prime-src-%d-%s-%s"' % (primeid, CUTOUT_RENDER_VERSION, day)
    headers = {'Cache-Control': 'private, max-age=0, must-revalidate', 'ETag': etag}
    if request.headers.get('If-None-Match') == etag:
        return Response(status=304, headers=headers)
    try:
        png = _cached_prime_source_png(primeid, day)
    except KeyError:
        abort(404)
    return Response(png, mimetype='image/png', headers=headers)


# ---------------------------------------------------------------------------
# PRIME source viewer: interactive LC data + cutouts strip   [prime_source_data]
# ---------------------------------------------------------------------------
@app.route('/prime/source_data/<int:primeid>.json')
@login_required
def prime_source_data(primeid):
    try:
        data = surveys.prime_source_lc_data(SURVEYS['prime'], primeid)
    except KeyError:
        abort(404)
    return jsonify(data)

@app.route('/prime/source_fp/<int:primeid>.csv')
@login_required
def prime_source_fp(primeid):
    try:
        name, csv_text = surveys.prime_source_fp_csv(SURVEYS['prime'], primeid)
    except KeyError:
        abort(404)
    safe = ''.join(c if (c.isalnum() or c in '._-') else '_'
                   for c in str(name)) or ('prime%d' % primeid)
    return Response(csv_text, mimetype='text/csv',
        headers={'Content-Disposition':
            'attachment; filename="%s_prime_fp.csv"' % safe})

@lru_cache(maxsize=512)
def _cached_prime_cutouts_png(primeid, day):
    return surveys.prime_render_source_cutouts_png(SURVEYS['prime'], primeid)

@app.route('/prime/source_cutouts/<int:primeid>.png')
@login_required
def prime_source_cutouts(primeid):
    day = _render_day()
    etag = '"prime-cut-%d-%s-%s"' % (primeid, CUTOUT_RENDER_VERSION, day)
    headers = {'Cache-Control': 'private, max-age=0, must-revalidate', 'ETag': etag}
    if request.headers.get('If-None-Match') == etag:
        return Response(status=304, headers=headers)
    try:
        png = _cached_prime_cutouts_png(primeid, day)
    except KeyError:
        abort(404)
    return Response(png, mimetype='image/png', headers=headers)


# ---------------------------------------------------------------------------
# Search index
# ---------------------------------------------------------------------------
@app.route('/wise_search')
@login_required
def search_index():
    return render_template('wise_search.html', user=session['user'],
        server=socket.gethostname(), timenow=now_str(), defpardict=defsearch)


# ---------------------------------------------------------------------------
# Parameter definitions page (shared across all scans)
# ---------------------------------------------------------------------------
PARAM_DEFS = [
    ('Epoch', "Single epoch only: SQL filters epochid = N (exact match, NOT cumulative / <=)."),
    ('RB score (lo/hi)', "Real-bogus classifier score; keeps rbscore between lo and hi (typ. 0-1)."),
    ('NMatches', "Minimum nmatches -- number of matched detections of the source PRIOR to the epoch."),
    ('Scorr peak', "Minimum scorr_peak: peak of the Scorr (Zackay-Ofek-Gal-Yam matched-filter) detection statistic at the candidate, ~ detection S/N in sigma."),
    ('|Gal. lat| (lo/hi)', "Absolute Galactic latitude band: abs(f.gallat) in [lo, hi) degrees."),
    ('Age (lo/hi)', "Days since first detection: mjd - firstdet, strict > lo AND < hi."),
    ('Epoch brightness (lo/hi, YSO)', "psf_mag >= lo AND psf_mag < hi of the detection at the scanned epoch (smaller mag = brighter; single epoch; inclusive-low / exclusive-high so ranges tile without gaps or overlaps)."),
    ('Bright-star rejection', "distnearbrstar > brdistlim, and each WISE neighbour must be far (wdist > brdistlim) OR faint (wXmag > brmaglim)."),
    ('Hostless WISE dist', "All three WISE neighbours farther than this (arcsec)."),
    ('Max WISE dist / Amplitude (hosted)', "Nearest WISE source within hwdist arcsec AND brightening vs AllWISE >= hwamp mag."),
    ('NED match dist', "Source-to-galaxy separation (arcsec) from the NED crossmatch."),
    ('Bright <= ... for over (YSO)', "W1 stayed brighter than durmag mag for a continuous span longer than minduration years."),
    ('W1-W2 colour (lo/hi, YSO)', "Per-epoch median W1-W2 colour within [lo, hi]."),
    ('Skip', "Pagination offset: number of (deduplicated) sources to skip."),
]

@app.route('/definitions')
@login_required
def definitions_page():
    return render_template('definitions.html', user=session['user'],
        server=socket.gethostname(), timenow=now_str(), param_defs=PARAM_DEFS)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route('/healthz')
def healthz():
    checks = {}
    ok = True
    try:
        app.jinja_env.get_template('login.html')
        checks['templates'] = 'ok'
    except Exception:
        checks['templates'] = 'fail'
        ok = False
    for key, survey in SURVEYS.items():
        try:
            conn, cur = survey.open_cursor()
            try:
                cur.execute('SELECT 1;')
                cur.fetchone()
            finally:
                survey.close_cursor(conn, cur)
            checks['db:%s' % key] = 'ok'
        except KeyError:
            checks['db:%s' % key] = 'unconfigured'
        except Exception:
            checks['db:%s' % key] = 'fail'
            ok = False
    return jsonify(status='ok' if ok else 'degraded', checks=checks), (200 if ok else 503)

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
@app.route('/')
def main():
    if not session.get('logged_in'):
        return render_template('login.html')
    return render_template('wise_scan.html', user=session['user'],
        server=socket.gethostname(), timenow=datetime.now(UTC).strftime('%Y-%m-%d'))

# ---------------------------------------------------------------------------
# CLI: python app.py adduser <name>
# ---------------------------------------------------------------------------
def _cli():
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == 'adduser':
        name = sys.argv[2]
        pw = getpass.getpass('Password for %s: ' % name)
        users = load_users()
        users[name] = generate_password_hash(pw)
        with open(USERS_FILE, 'w') as fh:
            json.dump(users, fh, indent=2)
        os.chmod(USERS_FILE, 0o600)
        print('Stored hashed credentials for %s in %s' % (name, USERS_FILE))
        return True
    return False

if __name__ == '__main__':
    if not _cli():
        app.run(host='127.0.0.1', port=5000, threaded=True, debug=False)

# [patch] wtp-bugfix-20260717 applied

# [patch] wtp-nightly-window-20260717 applied
