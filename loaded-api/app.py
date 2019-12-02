#!flask/bin/python
from flask import Flask
from flask import jsonify
from flask import abort
from flask import make_response
from flask import request
from flask import url_for

#Import Dylan's model
import sys
sys.path.append("..")
import run_model_on_article

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


@app.route('/api/v2.0/apiget', methods=['GET'])
def call_with_hardcode():
	hardcoded_string = "{'body': 'she had sexy cleavage and a shrill voice as she talked about domestic affairs'}"
	return jsonify(apiget(hardcoded_string))

@app.route('/api/v2.0/post', methods=['POST'])
def handle_post_request():
	json = request.json #This works! A field called 'body' contains the entire text as a string
	#return jsonify(json) --- this would print out the json payload
	return jsonify(run_model_on_article.apiget(json)) # Not tested - would depend on functionality of the apiget function

#Old API below

@app.route('/api/v1.0/words/', methods=['GET'])
def get_words():
    return jsonify({'words': list(female_words)})

@app.route('/api/v1.0/words/<string:word>', methods=['GET'])
def get_word(word):
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
