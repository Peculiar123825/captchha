from flask import Flask, render_template, request, session, redirect, url_for
import random
import string
import os
from captcha.image import ImageCaptcha
from security import check_rate_limit

app = Flask(__name__)
app.secret_key = "captcha_secret_key_demo"

CHARSET = string.ascii_uppercase + string.digits
CAPTCHA_LENGTH = 4
CAPTCHA_DIR = os.path.join("static", "captchas")
os.makedirs(CAPTCHA_DIR, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["username"] = request.form.get("username", "student")
        return redirect(url_for("captcha_page"))
    return render_template("login.html")


@app.route("/captcha", methods=["GET"])
def captcha_page():
    text = ''.join(random.choices(CHARSET, k=CAPTCHA_LENGTH))
    session["captcha_answer"] = text
    image_gen = ImageCaptcha(width=160, height=60)
    filename = "current_captcha.png"
    path = os.path.join(CAPTCHA_DIR, filename)
    image_gen.write(text, path)
    return render_template("captcha.html", image_file=filename)


@app.route("/verify", methods=["POST"])
def verify():
    ip = request.remote_addr
    allowed, count = check_rate_limit(ip)

    if not allowed:
        return render_template("blocked.html", attempts=count)

    user_answer = request.form.get("answer", "").strip().upper()
    correct_answer = session.get("captcha_answer", "")

    if user_answer == correct_answer:
        return render_template("result.html", success=True, answer=correct_answer)
    else:
        return render_template("result.html", success=False, answer=correct_answer)


@app.route("/simulate_bot")
def simulate_bot():
    """
    Fires several rapid, simulated attempts from this session to
    demonstrate how the rate limiting module reacts to bot-like behaviour.
    """
    ip = request.remote_addr
    last_count = 0
    for _ in range(8):
        allowed, count = check_rate_limit(ip)
        last_count = count
        if not allowed:
            break
    return render_template("blocked.html", attempts=last_count, simulated=True)


if __name__ == "__main__":
    app.run(debug=True)
