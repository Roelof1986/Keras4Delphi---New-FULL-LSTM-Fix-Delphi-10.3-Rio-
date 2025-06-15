# Keras4Delphi---New-FULL-LSTM-Fix-Delphi-10.3-Rio-
A complete fix for stateful=True, return_sequences=True and batch_input_shape behaviour in Keras4Delphi. Enables reliable training of stateful LSTM models with static batch sizes and proper parameter handling.

1.  Download file "Keras.pas" and replace the old Keras.pas with the new version
2.  Make sure you are using the updated/debugged Keras4Delphi (in repository: Keras4Delphi---debugged-Delphi-10.3-Rio-)
3.  Download and run the example code: DoubleLayer_LSTM_test


# Keras4Delphi Stateful LSTM Fix

This repository contains a tested and working fix for using `stateful=True` in LSTM layers within [Keras4Delphi](https://github.com/Pigrecos/Keras4Delphi).

## ✅ What it fixes

- Prevents the common error: `ValueError: When using 'stateful=True' in a RNN, the batch size must be static`
- Enforces correct `batch_input_shape=(batch_size, timesteps, features)` handling
- Ensures Python receives `stateful=True` and `return_sequences=True` without loss of information

## 🔧 How it works

- Uses `class var` configuration for global LSTM defaults
- Automatically injects required keyword arguments via a modified `Instantiate` method
- Avoids fragile parameter passing through `.SetItem(...)`

## 💡 Why it's useful

Keras4Delphi is a powerful bridge between Delphi and Python's deep learning stack.  
But until now, configuring stateful LSTM layers was unreliable and prone to shape errors.  
This fix makes stateful sequence modeling stable, predictable, and production-ready.

## ✨ Highlights

- Compatible with TensorFlow 2.19 and Delphi 10.3/10.4
- Supports 2-layer LSTM stacks with controlled sequence output
- No more `None, batch, time, features` shape mismatches

## 📂 What's included

- Modified `Instantiate` logic (Delphi)
- Class-wide control via `TBase.UseStateful`, `UseReturnSequences`, `UseBatchInputShape`
- Tested example with correct loss convergence over 500 epochs

## 📜 License

MIT
