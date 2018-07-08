import sys
import os
import base64
from io import BytesIO
from flask import Flask, request, render_template, json
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import neural_network
n = None
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("./index.html")

@app.route("/query", methods=["POST"])
def query():
    if request.method == "POST":
        # TODO: 받은 이미지의 투명한 영역 흰 배경 채우기
        background = Image.new(mode="RGB", size=(28, 28), color=255)
        img = Image.open(BytesIO(base64.b64decode(request.json["data"])))
        img = img.resize((28, 28), Image.LANCZOS)
        # result = Image.merge("RGB", (background, img))
        img.save('convert.png')
        img.close()
        background.close()
    return json.dumps({"status": 200})

if __name__ == "__main__":
    print("신경망 학습이 끝난 후 서버가 실행 됩니다.")
    # n = neural_network.start()
    app.run(debug=True)