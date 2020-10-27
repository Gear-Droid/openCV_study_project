# import telebot
import datetime
from celery import Celery


app = Celery('tg_warning', broker='redis://localhost')


@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Calls remind_me() every 40 seconds
    sender.add_periodic_task(40.0, remind_me.s('world'), expires=10)


@app.task
def remind_me():
    hours = int(str(datetime.datetime.now().time())[0:2])
    minutes = int(str(datetime.datetime.now().time())[3:5])

    if hours == 17:
        if minutes == 50:
            print('ЖЕЛТЫЙ УРОВЕНЬ ТРЕВОГИ!')

    if hours == 17:
        if minutes == 57:
            print('КРАСНЫЙ УРОВЕНЬ ТРЕВОГИ!')

    print('{} : {}'.format(hours, minutes))

"""
bot = telebot.TeleBot(config.token)


if __name__ == '__main__':
    bot.polling(none_stop=True)
"""
