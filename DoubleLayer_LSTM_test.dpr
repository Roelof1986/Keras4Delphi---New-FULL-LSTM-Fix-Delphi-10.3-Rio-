program DoubleLayer_LSTM_test;

uses
  Keras,
  Keras.Layers,
  Keras.Models,
  System.Types,
  np.Base,
  np.Api,
  np.Utils,
  np.Models,
  PythonEngine,
  Python.Utils,
  System.IOUtils,
  SysUtils,
  Keras.PreProcessing,
  Windows,
  System.Rtti, System.Generics.Collections, System.TypInfo;

(*const
  batch_size = 2;
  timesteps  = 4;
  features   = 2;
  output_units = 1;  // b.v. 1 voor regressie of classificatie  *)

var
x, y, x_test, y_test : TNDarray;

y_out2 : Variant;

xtestarray, ytestarray : TArray<Double>;

i : Longint;

var batchShape : Tnp_shape;

lstm1, lstm2, dense : TBaseLayer;

begin

  TNumPy.Init(True);

  // ---

// Maak dummy input: x van vorm [2, 4, 2] (batch_size = 2, timesteps = 4, features = 2)

SetLength(xtestarray, 2 * 4 * 2);
for i := 0 to High(xtestarray) do
  xtestarray[i] := Random;
x := TNDArray(TNumPy.npArray<Double>(xtestarray).reshape([2, 4, 2]));

// Maak dummy output: y van vorm [2, 1] (batch_size = 2, output_units = 1)

SetLength(ytestarray, 2);
for i := 0 to High(ytestarray) do
  ytestarray[i] := Random;
y := TNDArray(TNumPy.npArray<Double>(ytestarray).reshape([2, 1]));

  // ---

  var model := TSequential.Create;

  // Eerste LSTM-layer (stateful, return_sequences, met batch_input_shape)

  TBase.UseStateful := True;
  TBase.UseReturnSequences := True;
  TBase.UseBatchInputShape := [2, 4, 2];

  {var} lstm1 := TLSTM.Create(32, 'sigmoid');

  model.Add(lstm1);

  // Tweede LSTM-layer (stateful, ontvangt sequentie van vorige laag)

  TBase.UseStateful := False;
  TBase.UseReturnSequences := False;
  TBase.UseBatchInputShape := [2, 4, 32];

  {var} lstm2 := TLSTM.Create(32, 'sigmoid');

  model.Add(lstm2);

  // Output-laag (bijv. Dense voor eindresultaat)
  model.Add( TDense.Create({output_units}1, 'sigmoid') );

  model.Compile('adam', 'mse');    // compileer met een optimizer en loss naar keuze
var batch_size2 : Integer := 2;
var history: THistory := model.Fit(x, y, @batch_size2, {10}500,1);
  model.Summary;                  // toon modeloverzicht (optioneel)

// Train met dummydata

// Voorspelling

{var} y_test := model.Predict(x);
{var} y_out2 := y_test.ToDoubleArray;
for i := 0 to 1 do
  Writeln('Predictie[', i, ']: ', y_out2[i]{:1:4});

// Opslaan model

//TFile.WriteAllText('model.json', model.ToJson);
//model.SaveWeight('model.h5');

// Einde
Readln;

end.
