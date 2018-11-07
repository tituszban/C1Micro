from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, concatenate
from keras.optimizers import Adam

def gen_model(lanes):

    input_info = Input(shape=(7,), name="inp_info")
    input_gamestate = Input(shape=(lanes,3), name="inp_gs")

    info = Dense(20, activation='relu', name="info_d1")(input_info)
    info = Dense(20, activation='relu', name="info_d2")(info)

    gs = Conv1D(kernel_size=1, filters=16, name="gs_c1")(input_gamestate)
    gs = Conv1D(kernel_size=1, filters=8, name="gs_c2")(gs)
    gs_flat = Flatten(name="gs_flat")(gs)

    joined = concatenate([info, gs_flat])
    joined = Dense(100, activation='relu', name="join_d1")(joined)

    action = Dense(20, activation='relu', name="out_a_pre")(joined)
    out_action = Dense(3, activation='softmax', name="out_a")(action)
    placement = Dense(40, activation='relu', name="out_p_pre")(joined)
    out_placement = Dense(lanes, activation='softmax', name="out_p")(placement)

    model = Model(input=[input_info, input_gamestate], outputs=[out_action, out_placement])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    print(model.summary())
    return model