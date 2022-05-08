import pandas as pd
import Predict
import DataCollection
from telegram.ext import Updater,CommandHandler,MessageHandler,Filters,ConversationHandler

API = "5295828222:AAFERy5DfBtJN736B9Xz-YpyGhlqc7Wg8fg"

PREDICTION = 0

def prediction_instructions(update,context):
    update.message.reply_text("Type in your stock Tickers:")
    update.message.reply_text("For example:AAPL DIS SEA")
    return PREDICTION

def predictions(update, context):
    text = update.message.text
    splitted = text.split(" ")
    if ("".join(splitted)).isalnum():
        update.message.reply_text("Analysing Stock's Performance!")
        update.message.reply_text(f"Estimated Time: {len(splitted)*10}s")
        df = DataCollection.full_run(splitted,"",True,True)
        if isinstance(df,pd.DataFrame):
            pred = Predict.make_predictions_df(df)
            print(pred)
            update.message.reply_text(pred)
        else:
            update.message.reply_text(df)
    else:
        update.message.reply_text("Only letters and numbers!")

def help_command(update, context):
    update.message.reply_text("Here's the list of commands!\n/start\n/help\n/predict")

def cancel(update, context):
    """Cancels and ends the conversation."""
    update.message.reply_text('Operation Cancelled!')
    return ConversationHandler.END

def error(update, context):
    print(f"Update {update} caused error {context.error}")

def main():
    updater = Updater(API, use_context=True)
    dp = updater.dispatcher

    predict_conv_handler = ConversationHandler(
    entry_points=[CommandHandler("predict",prediction_instructions)],
    states={
        PREDICTION: [MessageHandler(Filters.text, predictions)]
        },
    fallbacks=[CommandHandler('cancel', cancel)])

    dp.add_handler(predict_conv_handler)
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_error_handler(error)

    #start_polling(20) = bot will wait 20 seconds before checking for the next user input
    updater.start_polling()

    #updater.idle allows bot to keep running
    updater.idle()

main()