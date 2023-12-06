from flask import Flask , render_template,request,jsonify

#from chat import test_run
from  bot_webscrap import generate_response

app= Flask(__name__)


@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text=request.get_json().get("message")
    #response=test_run(text)
    response=generate_response(text)
    message={"answer":response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)