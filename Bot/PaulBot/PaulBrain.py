from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, concatenate
from keras.optimizers import Adam, SGD


def gen_model(lanes):
    input_info = Input(shape=(7,), name="inp_info")
    input_gamestate = Input(shape=(lanes, 3), name="inp_gs")

    info = Dense(20, activation='relu', name="info_d1", kernel_initializer="normal")(input_info)
    info = Dense(20, activation='relu', name="info_d2", kernel_initializer="normal")(info)

    gs = Conv1D(kernel_size=1, filters=16, name="gs_c1", kernel_initializer="normal")(input_gamestate)
    gs = Conv1D(kernel_size=1, filters=8, name="gs_c2", kernel_initializer="normal")(gs)
    gs_flat = Flatten(name="gs_flat")(gs)

    joined = concatenate([info, gs_flat])
    joined = Dense(100, activation='relu', name="join_d1", kernel_initializer="normal")(joined)

    action = Dense(20, activation='relu', name="out_a_pre", kernel_initializer="normal")(joined)
    out_action = Dense(3, name="out_a", activation="linear", kernel_initializer="normal")(action)  # 277
    placement = Dense(40, activation='relu', name="out_p_pre", kernel_initializer="normal")(joined)
    out_placement = Dense(lanes, name="out_p", activation="linear", kernel_initializer="normal")(placement)

    model = Model(inputs=[input_info, input_gamestate], outputs=[out_action, out_placement])
    model.compile(loss='mse', optimizer=Adam(lr=1e-6), metrics=['accuracy'])
    #Adam(lr=1e-6)

    # print(model.summary())
    return model
