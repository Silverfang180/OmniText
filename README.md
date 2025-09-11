# OmniText: The All-in-One NLP Text Analysis Tool 📝
## 💡 Overview
OmniText is a comprehensive and intuitive web application built with Streamlit for effortless Natural Language Processing (NLP). It provides a complete suite of tools to analyze, summarize, and understand any body of text. Whether you need to quickly grasp the main points of a long document or extract key information from a web article, OmniText has you covered.

## ✨ Features
1. **Intelligent Text Summarization**: Condense lengthy articles into a few key sentences. Choose from a variety of advanced algorithms, including LSA, LexRank, Luhn, and TextRank, and customize the summary length to fit your needs.

2. **Named Entity Recognition (NER):** Automatically identify and categorize important entities within your text, such as PERSONS, ORGANIZATIONS, LOCATIONS, and DATES. A visualizer highlights these entities, and a frequency chart shows the most common ones.

3. **URL Text Extraction:** Simply paste a URL, and OmniText will scrape the main content from the web page for analysis, making it easy to work with online articles and blog posts.

4. **Sentiment Analysis:** Get an overall sentiment score for your text, classifying it as positive, negative, or neutral.

5. **Word Cloud & Frequency Analysis:** Visualize the most frequently used words in your text with a dynamic word cloud and a bar chart, helping you quickly identify the main themes.

## 🛠️ Technologies Used
- **Streamlit**: The primary framework for building the interactive and user-friendly web interface.

- **spaCy**: A powerful library for high-performance Named Entity Recognition.

- **sumy**: A versatile library that provides various unsupervised text summarization algorithms.

- **TextBlob**: Used for performing simple and effective sentiment analysis.

- **Requests & BeautifulSoup**: Essential libraries for extracting text from web pages.

- **WordCloud & Matplotlib**: For generating insightful data visualizations.

## 🚀 How to Run Locally
 **1. Clone the Repository**

Bash

git clone https://github.com/Silverfang180/omnitext.git
cd omnitext

 **2. Create and Activate a Virtual Environment**

Bash

python -m venv venv
  **3. Install Required Libraries**
Bash

pip install -r requirements.txt

 **4. Download the spaCy Model**
Bash

python -m spacy download en_core_web_sm

 **5. Run the Streamlit App**

Bash

streamlit run app.py

The application will automatically open in your default web browser.

## 🤝 Contributing
Contributions are highly welcome! If you have ideas for new features, improvements, or bug fixes, please feel free to open an issue or submit a pull request.

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Samples
<img width="940" height="458" alt="image" src="https://github.com/user-attachments/assets/749f2088-d088-4170-98f0-d11b7bec0593" />

- From the above image, we are going to perform Summarization feature.

<img width="941" height="463" alt="image" src="https://github.com/user-attachments/assets/e239f256-fe34-4cd8-bfa5-93964f50986c" />

- This is an output of Summarization.

<img width="940" height="462" alt="image" src="https://github.com/user-attachments/assets/7cc490e0-630c-4883-9535-4ba7fd654c6e" />

- In this image, we perform NER(Name Entity Recognition) feature.

