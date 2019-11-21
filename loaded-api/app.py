#!flask/bin/python
from flask import Flask
from flask import jsonify
from flask import abort
from flask import make_response
from flask import request
from flask import url_for

app = Flask(__name__)

#Defines corpus of words --- maybe in the future we can abstract this away
#Currently only has 2 dummy female words because I'm assuming that we will replace this with a proper model
female_words={
"shrill",
"wife"
}


@app.route('/')
def index():
    return "The Loaded Words API is up and running! Let's make sure your words are not gender biased. :)"

#We only need a GET because the basic functionality required is to ping the API for whether a word is loaded or not. (We don't want a way to modify the underlying data, so PUT/POST are not necessary.)

@app.route('/api/v1.0/words/', methods=['GET'])
def get_tasks():
    return jsonify({'words': list(female_words)})

@app.route('/api/v1.0/words/<string:word>', methods=['GET'])
def get_task(word):
	if word in female_words:
		return jsonify({'loaded-score': 1})
	else:
		return jsonify({'loaded-score': 0})

#Gracefully handles a 404 error
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(debug=True)
