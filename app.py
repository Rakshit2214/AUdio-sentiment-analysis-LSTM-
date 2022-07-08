import fileinput
import speech_recognition as sr
from flask import Flask, render_template, request, flash,redirect
from flask_wtf import FlaskForm
from wtforms import FileField
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pickle
import warnings
app = Flask(__name__)
app.secret_key = "Rakshit_app2214"

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

class MyForm(FlaskForm):
    image = FileField('image')




@app.route("/", methods=['POST', 'GET'])
@app.route("/analyze",methods=['POST', 'GET'])
def analyzer():
	transcript=""
	form = MyForm()


	if request.method == 'POST':
		print("data received")

		if 'file' not in request.files:
			return redirect(request.url)
		file = request.files["file"]
		if file.filename == "":
			return redireect(request.url)
		if file:
			r = sr.Recognizer()
			file_audio = sr.AudioFile(file)

			with file_audio as source:
				audio = r.record(source)
			try:
				text = r.recognize_google(audio)
				
				validation_sentence = [text]
				validation_sentence_tokened = tokenizer.texts_to_sequences(validation_sentence)
				validation_sentence_padded = pad_sequences(validation_sentence_tokened, maxlen=128, truncating='post',
														   padding='post')

				transcript=str(validation_sentence[0])+"Positivity Index: "+ str(model.predict(validation_sentence_padded)[0][0])



			except:
				transcript='sorry.. try again'

			return render_template("index.html", transcript=transcript)

	return render_template("audio_file.html", form=form)




if __name__ == '__main__':
	app.run()
