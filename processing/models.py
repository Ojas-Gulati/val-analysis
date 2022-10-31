def makeLSTMmodel(): #AR3
    plantLocations_input_um = keras.Input(shape=(2,), name="plantLocations")
    playerLocations_input_um = keras.Input(shape=(None, 2), name="playerLocations")
    playerTeams_input_um = keras.Input(shape=(None, 1), name="playerTeams")
    
    mask = keras.layers.Masking()
    plantLocations_input = mask(plantLocations_input_um)
    playerLocations_input = mask(playerLocations_input_um)
    playerTeams_input = mask(playerTeams_input_um)

    denseLayer1 = keras.layers.Dense(8, activation='relu')
    denseLayer2 = keras.layers.Dense(4, activation='relu')

    def locationEncode(x):
        return denseLayer2(denseLayer1(x))

    plant_location_encode = locationEncode(plantLocations_input)
    players_location_encode = locationEncode(playerLocations_input)
    players_and_teams = keras.layers.Concatenate()([players_location_encode, playerTeams_input])

    lstm_size = 128
    lstm_layer = keras.layers.LSTM(lstm_size, return_sequences=True)(players_and_teams, initial_state=[keras.layers.Dense(lstm_size)(plant_location_encode), keras.layers.Dense(lstm_size)(plant_location_encode)])
    lstm_layer_2 = keras.layers.LSTM(64)(lstm_layer)
    fin2_layer = keras.layers.Dense(16, activation='relu')(lstm_layer_2)
    final_layer = keras.layers.Dense(1, activation='sigmoid')(fin2_layer)

    return keras.Model(inputs=[plantLocations_input_um, playerLocations_input_um, playerTeams_input_um], outputs=final_layer)

def makeLSTMmodel(): # AR2
    plantLocations_input_um = keras.Input(shape=(2,), name="plantLocations")
    playerLocations_input_um = keras.Input(shape=(None, 2), name="playerLocations")
    playerTeams_input_um = keras.Input(shape=(None, 1), name="playerTeams")
    
    mask = keras.layers.Masking()
    plantLocations_input = mask(plantLocations_input_um)
    playerLocations_input = mask(playerLocations_input_um)
    playerTeams_input = mask(playerTeams_input_um)

    denseLayer1 = keras.layers.Dense(8, activation='relu')
    denseLayer2 = keras.layers.Dense(4, activation='relu')

    def locationEncode(x):
        return denseLayer2(denseLayer1(x))

    plant_location_encode = locationEncode(plantLocations_input)
    players_location_encode = locationEncode(playerLocations_input)
    players_and_teams = keras.layers.Concatenate()([players_location_encode, playerTeams_input])

    lstm_size = 64
    lstm_layer = keras.layers.LSTM(lstm_size)(players_and_teams, initial_state=[keras.layers.Dense(lstm_size)(plant_location_encode), keras.layers.Dense(lstm_size)(plant_location_encode)])
    fin2_layer = keras.layers.Dense(16, activation='relu')(lstm_layer)
    final_layer = keras.layers.Dense(1, activation='sigmoid')(fin2_layer)

    return keras.Model(inputs=[plantLocations_input_um, playerLocations_input_um, playerTeams_input_um], outputs=final_layer)

def makeLSTMmodel(): #AR4, and FIN
    plantLocations_input_um = keras.Input(shape=(2,), name="plantLocations")
    playerLocations_input_um = keras.Input(shape=(None, 2), name="playerLocations")
    playerTeams_input_um = keras.Input(shape=(None, 1), name="playerTeams")
    
    mask = keras.layers.Masking()
    plantLocations_input = mask(plantLocations_input_um)
    playerLocations_input = mask(playerLocations_input_um)
    playerTeams_input = mask(playerTeams_input_um)

    denseLayer1 = keras.layers.Dense(16, activation='relu')
    denseLayer2 = keras.layers.Dense(8, activation='relu')

    def locationEncode(x):
        return denseLayer2(denseLayer1(x))

    plant_location_encode = locationEncode(plantLocations_input)
    players_location_encode = locationEncode(playerLocations_input)
    players_and_teams = keras.layers.Concatenate()([players_location_encode, playerTeams_input])

    lstm_size = 64
    lstm_layer = keras.layers.LSTM(lstm_size)(players_and_teams, initial_state=[keras.layers.Dense(lstm_size)(plant_location_encode), keras.layers.Dense(lstm_size)(plant_location_encode)])
    # lstm_layer_2 = keras.layers.LSTM(64)(lstm_layer)
    fin2_layer = keras.layers.Dense(16, activation='relu')(lstm_layer)
    final_layer = keras.layers.Dense(1, activation='sigmoid')(fin2_layer)

    return keras.Model(inputs=[plantLocations_input_um, playerLocations_input_um, playerTeams_input_um], outputs=final_layer)

def makeLSTMmodel(): #AR5
    plantLocations_input_um = keras.Input(shape=(2,), name="plantLocations")
    playerLocations_input_um = keras.Input(shape=(None, 2), name="playerLocations")
    playerTeams_input_um = keras.Input(shape=(None, 1), name="playerTeams")
    
    mask = keras.layers.Masking()
    plantLocations_input = mask(plantLocations_input_um)
    playerLocations_input = mask(playerLocations_input_um)
    playerTeams_input = mask(playerTeams_input_um)

    denseLayer1 = keras.layers.Dense(32, activation='relu')
    denseLayer2 = keras.layers.Dense(16, activation='relu')

    def locationEncode(x):
        return denseLayer2(denseLayer1(x))

    plant_location_encode = locationEncode(plantLocations_input)
    players_location_encode = locationEncode(playerLocations_input)
    players_and_teams = keras.layers.Concatenate()([players_location_encode, playerTeams_input])

    lstm_size = 64
    lstm_layer = keras.layers.LSTM(lstm_size)(players_and_teams, initial_state=[keras.layers.Dense(lstm_size)(plant_location_encode), keras.layers.Dense(lstm_size)(plant_location_encode)])
    # lstm_layer_2 = keras.layers.LSTM(64)(lstm_layer)
    fin2_layer = keras.layers.Dense(32, activation='relu')(lstm_layer)
    final_layer = keras.layers.Dense(1, activation='sigmoid')(fin2_layer)

    return keras.Model(inputs=[plantLocations_input_um, playerLocations_input_um, playerTeams_input_um], outputs=final_layer)