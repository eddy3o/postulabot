from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from flaskr.auth import login_required
from flaskr.db import get_db

bp = Blueprint('bot', __name__, url_prefix='/bot')

@bp.route('/bot')
@login_required
def bot():
    db = get_db()
    user_id = g.user['id']
    user = db.execute(
        'SELECT * FROM user WHERE id = ?', (user_id,)
    ).fetchone()
    return render_template('bot/postulabot.html', user=user)