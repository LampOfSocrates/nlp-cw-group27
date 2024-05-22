import requests
import pandas as pd
import time
import matplotlib.pyplot as plt

url = "http://127.0.0.1:5000"  # URL for NER requests
train_url = "http://127.0.0.1:5000/train"  # URL for training requests
data = {"text": "Sample text for NER"}

response_times = []

# Measure training time
train_response = requests.post(train_url)
training_time = train_response.json()['training_time']
print(f'Training time: {training_time} seconds')

# Send multiple requests to measure response time
for _ in range(100):  # Adjust the number of requests as needed
    start_time = time.time()
    response = requests.post(url, json=data)
    response_times.append(time.time() - start_time)

# Create a DataFrame and generate descriptive statistics
df = pd.DataFrame(response_times, columns=["response_time"])
print(df.describe())

# Generate a histogram of response times
df.plot(kind='hist', title='Response Time Distribution')
plt.xlabel('Response Time (seconds)')
plt.ylabel('Frequency')
plt.show()



