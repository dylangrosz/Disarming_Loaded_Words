API README

What I followed to build this:
https://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask


# How to run the API locally

To run the API locally:
chmod a+x app.py
./app.py

Alternatively, run `python3 app.py`.


# How to run the API on the Google Docs Plugin

The local API address from localhost needs to be linked to a non-localhost source.
To do this, I used ngrok,then did the following steps:

1. Download ngrok [https://ngrok.com/download].
2. Run the API locally (see above).
3. Once it loads, run the command on your terminal, in the directory where you have ngrok, run: `./ngrok http 5000`

This forwards localhost:5000 an ngrok address. Copy/paste this address into the Google Docs plugin code to make the API call from Google Docs.


# How to use the API endpoint to get the contents of an article from Google Docs

The body of the document will be passed to the backend as a string, via an API post request.

The endpoint that handles this request is @app.route('/api/v2.0/post', methods=['POST']). We need a post request because GET requests do not have a request body.

After an API call is made in the frontend to http://localhost:5000/api/v2.0/post, the JSON payload is stored in the variable `request.json`. The format of the payload looks something like this:

{
	'body': "This is the contents of the article."
}

This variable should then get passed to the backend model.



# How to use the dummy API (the v1.0 endpoints)

After running, to view a list of all words in dictionary:
http://localhost:5000/api/v1.0/words/

After running, to get the score of a word:
http://localhost:5000/api/v1.0/words/<your_word>
