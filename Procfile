web: python app.py runserver
gunicorn app:application --preload -b 0.0.0.0:5000 

worker:  bundle exec rake jobs:work
