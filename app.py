from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

# Load the model and scaler
model = tf.keras.models.load_model(os.path.join('.', 'NutrientLevel.h5'))
scaler = joblib.load(os.path.join('.', 'scaler.pkl'))

# Define the result labels
pH = [None, 'Acidic', 'Slightly Acidic', 'Neutral', 'Slightly Alkaline', 'Alkaline']
EC_result = [None, 'Negligible', 'Harmless', 'Normal', 'Harmful', 'Very Harmful']
result = [None, "Very Low", "Low", "Moderate", "High", "Very High"]

def generate_fertilizer_suggestions(fertilizer_data):
    # Mapping of nutrient levels to fertilizer suggestions
    suggestions = []

    nutrient_recommendations = {
        "pH": {
            "Acidic": "Apply lime to increase pH.",
            "Slightly Acidic": "Consider adding lime, but monitor pH closely.",
            "Neutral": "No adjustment needed for pH.",
            "Slightly Alkaline": "Add organic matter or sulfur to slightly reduce pH.",
            "Alkaline": "Apply sulfur or aluminum sulfate to lower pH."
        },
        "EC": {
            "Negligible": "Add organic matter to improve soil conductivity.",
            "Harmless": "No adjustment needed for EC.",
            "Normal": "Soil salinity is balanced; maintain current practices.",
            "Harmful": "Consider flushing soil with water to reduce salinity.",
            "Very Harmful": "Use gypsum to mitigate high salinity."
        },
        "OC": {
            "Very Low": "Apply organic fertilizers like compost or manure.",
            "Low": "Increase organic matter through green manure or biofertilizers.",
            "Moderate": "Maintain organic content with regular composting.",
            "High": "Current organic matter is sufficient; maintain levels.",
            "Very High": "Monitor soil for nutrient imbalance due to excess organic matter."
        },
        "N": {
            "Very Low": "Apply urea or ammonium sulfate for nitrogen deficiency.",
            "Low": "Use nitrogen-rich fertilizers like ammonium nitrate.",
            "Moderate": "Consider foliar sprays for small adjustments.",
            "High": "Avoid additional nitrogen; risk of over-fertilization.",
            "Very High": "Reduce nitrogen input to prevent nutrient leaching."
        },
        "P": {
            "Very Low": "Apply single super phosphate (SSP) or bone meal.",
            "Low": "Use triple super phosphate (TSP) for phosphorus deficiency.",
            "Moderate": "Maintain phosphorus levels with balanced fertilizers.",
            "High": "Avoid adding phosphorus to prevent excess.",
            "Very High": "Monitor for phosphorus toxicity and reduce input."
        },
        "K": {
            "Very Low": "Add muriate of potash (MOP) or sulfate of potash (SOP).",
            "Low": "Use potassium sulfate to supplement potassium.",
            "Moderate": "Maintain potassium levels with balanced NPK fertilizers.",
            "High": "Avoid additional potassium application.",
            "Very High": "Reduce potassium to prevent toxicity."
        },
        "S": {
            "Very Low": "Apply gypsum or elemental sulfur to supplement sulfur.",
            "Low": "Use ammonium sulfate or magnesium sulfate to improve sulfur levels.",
            "Moderate": "Maintain current sulfur levels through regular fertilization.",
            "High": "Avoid additional sulfur inputs to prevent toxicity.",
            "Very High": "Monitor sulfur levels for potential nutrient imbalances."
        },
        "Zn": {
            "Very Low": "Apply zinc sulfate or zinc chelates to address zinc deficiency.",
            "Low": "Use foliar sprays of zinc-containing fertilizers.",
            "Moderate": "Maintain zinc levels with balanced micronutrient mixes.",
            "High": "Avoid excessive zinc inputs to prevent accumulation.",
            "Very High": "Monitor for zinc toxicity and reduce input."
        },
        "Fe": {
            "Very Low": "Use ferrous sulfate or iron chelates to address iron deficiency.",
            "Low": "Apply iron foliar sprays for faster uptake.",
            "Moderate": "Maintain current iron levels with micronutrient fertilizers.",
            "High": "Avoid additional iron to prevent over-application.",
            "Very High": "Monitor soil for iron toxicity; consider reducing inputs."
        },
        "Cu": {
            "Very Low": "Apply copper sulfate or chelated copper for deficiency.",
            "Low": "Use foliar sprays with copper-based fertilizers.",
            "Moderate": "Maintain current levels of copper with balanced micronutrients.",
            "High": "Avoid adding copper to prevent toxicity.",
            "Very High": "Monitor for copper toxicity; avoid further applications."
        },
        "Mn": {
            "Very Low": "Apply manganese sulfate or chelated manganese to correct deficiency.",
            "Low": "Use foliar sprays with manganese fertilizers.",
            "Moderate": "Maintain manganese levels with balanced fertilization.",
            "High": "Avoid adding manganese to prevent over-saturation.",
            "Very High": "Monitor for manganese toxicity and reduce input if needed."
        },
        "B": {
            "Very Low": "Use borax or boric acid to supplement boron.",
            "Low": "Apply boron-containing foliar sprays.",
            "Moderate": "Maintain boron levels with periodic soil amendments.",
            "High": "Avoid excessive boron application to prevent toxicity.",
            "Very High": "Monitor soil for boron toxicity; reduce further inputs."
        }
    }

    # Iterate over the fertilizers data to generate suggestions
    for nutrient, level in fertilizer_data.items():
        if nutrient in nutrient_recommendations and level in nutrient_recommendations[nutrient]:
            suggestions.append({"nutrient": nutrient, "suggestion": nutrient_recommendations[nutrient][level]})

    return suggestions

@app.route('/fertilizers', methods=['POST'])
def predict_fertilizers():
    try:
        # Parse JSON request data
        received_data = request.get_json()
        values = {item['nutrient']: item['value'] for item in received_data["values"]}
        print(values)
        
        # Prepare input for the model
        custom_input = np.array([
            [
                float(values["pH"]), float(values["EC"]), float(values["OC"]), float(values["N"]),
                float(values["P"]), float(values["K"]), float(values["S"]), float(values["Zn"]),
                float(values["Fe"]), float(values["Cu"]), float(values["Mn"]), float(values["B"])
            ]
        ])

        # Scale the input
        custom_input_scaled = scaler.transform(custom_input)

        # Make predictions
        custom_predictions = model.predict(custom_input_scaled)

        # Convert predictions to class labels
        custom_labels = [np.argmax(custom_predictions[i], axis=1) + 1 for i in range(12)]
        print(custom_labels)

        # Prepare response
        nutrient_levels = {
                "pH":  pH[custom_labels[0][0]],
                "EC" : EC_result[custom_labels[1][0]],
                "OC" : result[custom_labels[2][0]],
                "N" : result[custom_labels[3][0]],
                "P" : result[custom_labels[4][0]],
                "K" : result[custom_labels[5][0]],
                "S" : result[custom_labels[6][0]],
                "Zn" :result[ custom_labels[7][0]],
                "Fe" : result[custom_labels[8][0]],
                "Cu" : result[custom_labels[9][0]],
                "Mn" : result[custom_labels[10][0]],
                "B" : result[custom_labels[11][0]],
            }
        
        response = {
            "nutrient_levels": nutrient_levels,
            "nutrient_suggestions": generate_fertilizer_suggestions(nutrient_levels)
        }
        
        print(response)
        return jsonify(response)

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
