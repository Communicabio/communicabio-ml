import telegram
from telegram.ext import Updater
from telegram.error import NetworkError, Unauthorized
from telegram.ext import MessageHandler, Filters
from interactor import process_new_replic
import logging
TOKEN = "1079728001:AAElKzs3sokX7puQBnerJRbGyJ0acjETXL0"

def processer(bot, update):
    #print(update, context)
    # print(type(update), type(context))
    print(update.message.text)
    text = process_new_replic(update.message.from_user.id, update.message.text)
    bot.send_message(chat_id=update.effective_chat.id, text=text)

if __name__ == '__main__':
    # https://t.me/socks?server=95.85.18.95&port=8080&user=socksproxy&pass=telegramNeBolei
    domain = '104.244.77.254'
    port = '8080'
    REQUEST_KWARGS = {
    # "USERNAME:PASSWORD@" is optional, if you need authentication: # USERNAME:PASSWORD@
        'proxy_url': f'http://{domain}:{port}/',
        }
    updater = Updater(TOKEN, request_kwargs=REQUEST_KWARGS)
    dispatcher = updater.dispatcher
    echo_handler = MessageHandler(Filters.text | Filters.command, processer)
    dispatcher.add_handler(echo_handler)
    updater.start_polling()
