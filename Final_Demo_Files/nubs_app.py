#!/usr/bin/python3.6

from flask import Flask
from flask import render_template, url_for, redirect, request
from tinydb import TinyDB, Query
import json

app = Flask(__name__)
db = TinyDB('database.json')

@app.route('/')
def home():
	return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
	identifications = db.all()
	identifications.sort(key=lambda x: x['time'], reverse=True)
	return render_template('dashboard.html', identifications=identifications)

@app.route('/csi/new', methods=['POST'])
def new_csi():
	data = request.get_json()
	csi_person = data['person']
	csi_time = data['time']
	db.insert({'person': csi_person, 'time': csi_time})
	
	return redirect(url_for('dashboard'))

if __name__ == "__main__":
	app.run(debug=True)
