Basic documentation on how to use this thing:

What I followed to build this:
https://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask

To run the API locally:
chmod a+x app.py
./app.py

After running, to view a list of all words in dictionary:
http://localhost:5000/api/v1.0/words/

After running, to get the score of a word:
http://localhost:5000/api/v1.0/words/<your_word>
