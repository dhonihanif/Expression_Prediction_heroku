web: python app.py runserver
web: gunicorn app.wsgi:application --log-file - --log-level debug
python manage.py collectstatic --noinput
manage.py migrate

worker:  bundle exec rake jobs:work
