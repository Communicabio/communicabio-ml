import telegram
from telegram.error import NetworkError, Unauthorized
from interactor import process_new_replic
import logging
TOKEN = "1079728001:AAElKzs3sokX7puQBnerJRbGyJ0acjETXL0"

def echo(bot):
    """Echo the message the user sent."""
    global update_id
    # Request updates after the last update_id
    for update in bot.get_updates(offset=update_id, timeout=30):
        update_id = update.update_id + 1

        if update.message:  # your bot can receive updates without messages
            # Reply to the message'
            logging.info(update.message)
            from_id = update.message.from_user.id
            text = process_new_replic(from_id, update.message.text)
            update.message.reply_text(text)


if __name__ == '__main__':
    bot = telegram.Bot(TOKEN)
    try:
        update_id = bot.get_updates()[0].update_id
    except IndexError:
        update_id = None

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    while True:
        try:
            echo(bot)
        except NetworkError:
            sleep(1)
        except Unauthorized:
            # The user has removed or blocked the bot.
            update_id += 1
