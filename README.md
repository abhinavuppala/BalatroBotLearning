# BalatroBotLearning

## Running the Bot

As of now we don't have a trained model yet, but we can run the flush bot (which is hardcoded to only play flushes) to test the botting setup and ensure we are able to automatically play the game.

1. Install [Steamodded](https://github.com/Steamodded/smods/wiki/Installing-Steamodded-windows)
2. Clone the repository to your local machine
3. Install requirements with `pip install -r requirements.txt`, using Python 3.11
4. Ensure that the game path in `balatro_connection.start_balatro_instance` is where your Balatro game executable is located (You must already own balatro).
5. Run `flush_bot.py`, which should launch Balatro, and after around a 10 sec delay, will start playing the game automatically.

The balatrobot mod & API was developed by [besteon](https://github.com/besteon/balatrobot)