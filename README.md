# PulseVision-AI-Driven-Dosha-Imbalance-Detection-Using-Machine-Learning-for-Automated-Nadi-Pariksha
PulseVision is an AI-powered system that modernizes Ayurvedic Nadi Pariksha by analyzing pulse signals with machine learning to detect dosha imbalances (Vata, Pitta, Kapha), aiding in early health diagnosis and personalized wellness insights.
The project involves preprocessing raw pulse signals to remove noise, applying feature engineering methods to capture important time and frequency domain characteristics, and training multiple machine learning models to classify dosha types. By integrating signal processing with predictive modeling, PulseVision offers a data-driven approach to traditional diagnostics, promoting early detection of health issues and enabling personalized wellness recommendations. The project not only bridges ancient medical wisdom with modern AI technologies but also lays the groundwork for future expansion into real-time IoT-based monitoring and mobile health applications, making holistic healthcare more accessible, precise, and technology-driven.

**🧠 Project Highlights**
Pulse Signal Processing: Captures and pre-processes pulse waveforms for clean analysis.
Dosha Prediction: Utilizes machine learning models to predict dominant or imbalanced doshas.
Feature Engineering: Extracts time-domain and frequency-domain features to enhance model performance.
Model Comparison: Evaluates models based on accuracy, precision, recall, and F1 score.
Holistic Approach: Combines traditional Ayurvedic diagnostic methods with modern AI techniques.

**🚀 Technologies Used**
Python 3.x
Scikit-learn
Pandas, NumPy
Matplotlib, Seaborn
Signal Processing Libraries

**⚙️ How It Works**
Data Collection: Pulse signals are acquired using wearable sensors.
Preprocessing: Raw signals are cleaned and normalized.
Feature Extraction: Extract meaningful features like pulse rate, amplitude, frequency bands.
Model Training: ML models are trained and validated to detect dosha imbalances.
Prediction: The system predicts the primary or imbalanced dosha based on pulse characteristics.

**🎯 Future Improvements**
Real-time pulse signal acquisition using IoT devices.
Integration of deep learning models (CNNs/RNNs) for improved accuracy.
Mobile application for user-friendly health reports.
Personalized health recommendations based on dosha profile.

The dataset used for this project is a custom dataset of pulse signals. Due to privacy and size considerations, only a sample is provided in the repository. The full dataset follows the same structure and can be generated.

**Inputs:**
Users will enter the following physiological data for analysis:
Age (years)
Weight (kg)
Height (cm)
Blood Pressure (mmHg)
Temperature (°C)
Pulse Rate (beats per minute)

**Dosha Predictions:**
Based on the input data, the system predicts one or more of the following doshas or their combinations:
Vata: Often associated with light, dry, cold, and irregular qualities. Indicators may include a high pulse rate, low blood pressure, and low body temperature.
Pitta: Known for hot, sharp, and intense qualities. Indicators include high body temperature, high blood pressure, and moderate pulse rate.
Kapha: Linked to cool, moist, and heavy qualities. Indicators include low pulse rate, normal temperature, and higher weight.
Combination Doshas:
  Vata-Pitta: A mix of dry, cold, and hot qualities. Common in individuals who experience both nervous energy and intense emotional responses.
  Pitta-Kapha: A blend of sharp, intense qualities with cool, stable energy. May show characteristics of both fiery and heavy tendencies.
  Vata-Kapha: A combination of irregular, dry qualities with heavy, slow characteristics. Often seen in people who are anxious yet physically sluggish.

**Recommendations:**
The system provides customized recommendations depending on the predicted dosha(s), such as:
Vata Dosha: Grounding foods, warm environments, and stress-reducing activities.
Pitta Dosha: Cooling foods, relaxing practices, and stress management techniques.
Kapha Dosha: Energizing foods, regular physical activity, and mental stimulation.
Combination Doshas: Tailored advice to balance both doshas in the combination, promoting harmony through lifestyle adjustments and mindfulness practices.

**Yoga Pose Suggestions:**
For each dosha or combination of doshas, the system recommends a yoga pose designed to balance the individual’s energy:
