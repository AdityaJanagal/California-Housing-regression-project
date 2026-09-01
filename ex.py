
from flask import Flask, request, render_template

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application


# Home page
@app.route("/")
def index():
    return render_template("home.html")


# Prediction page
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    # When opening the prediction page
    if request.method == "GET":
        return render_template("index.html")

    # When submitting the form
    else:

        # Get input data from HTML form
        data = CustomData(
            MedInc=float(request.form.get("MedInc")),
            HouseAge=float(request.form.get("HouseAge")),
            AveRooms=float(request.form.get("AveRooms")),
            AveBedrms=float(request.form.get("AveBedrms")),
            Population=float(request.form.get("Population")),
            AveOccup=float(request.form.get("AveOccup")),
            Latitude=float(request.form.get("Latitude")),
            Longitude=float(request.form.get("Longitude"))
        )

        # Convert input into DataFrame
        pred_df = data.get_data_as_dataframe()

        print("Input Data:")
        print(pred_df)

        print("Before Prediction")

        # Prediction pipeline
        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)

        print("After Prediction")
        print("Prediction:", results)

        prediction = results[0] * 100000

        return render_template(
                 "index.html",
                results=round(prediction, 2)
            )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
