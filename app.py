import os
import base64
from io import BytesIO
from fastai import *
from fastai.vision import *
from flask import Flask, jsonify, request, render_template
from werkzeug.exceptions import BadRequest
#from hints import fact_finder


def evaluate_image(img):
    pred_class, pred_idx, outputs = trained_model.predict(img)
    return pred_class

def load_model():
    path = '/floyd/home'
    classes = ['prunus_spinosa', 'fraxinus_excelsior', 'salix_matsudana', 'celtis_occidentalis', 'syringa_vulgaris', 'sambucus_nigra', 'elaeagnus_angustifolia', 'aesculus_hippocastamon', 'laburnum_anagyroides', 'prunus_padus', 'fagus_sylvatica', 'liriodendron_tulipifera', 'euonymus_europaea', 'gymnocladus_dioicus', 'pyrus_communis', 'staphylea_pinnata', 'corylus_colurna', 'amelanchier_spicata', 'populus_alba', 'acer_pseudoplatanus', 'sorbus_intermedia', 'ailanthus_altissima', 'cercis_canadensis', 'armeniaca_vulgaris', 'viburnum_opulus', 'betula_pendula', 'ulmus_minor', 'salix_caprea', 'ptelea_trifoliata', 'cotinus_coggygria', 'acer_platanoides', 'ginkgo_biloba', 'ilex_aquifolium', 'populus_deltoides', 'ulmus_laevis', 'magnolia_denudata', 'populus_tremula', 'prunus_domestica', 'tilia_cordata', 'acer_campestre', 'corylus_avellana', 'quercus_rubra', 'acer_saccharinum', 'alnus_glutinosa', 'koelreuteria_paniculata', 'sorbus_torminalis', 'persica_vulgaris', 'salix_babylonica', 'quercus_petraea', 'sambucus_racemosa', 'sophora_japonica', 'rhamnus_cathartica', 'viburnum_lantana', 'acer_negundo', 'liquidambar_styraciflua', 'morus_alba', 'gleditsia_triacanthos', 'malus_domestica', 'ficus_carica', 'acer_tataricum', 'carpinus_betulus', 'tilia_tomentosa', 'robinia_pseudo-acacia', 'padus_avium', 'crataegus_sanguinea', 'frangula_alnus', 'acer_palmatum', 'chionanthus_virginicus', 'paulownia_tomentosa', 'catalpa_bignonioides', 'sorbus_aria', 'ulmus_glabra', 'populus_nigra', 'prunus_mahaleb', 'ulmus_pumila', 'maclura_pomifera', 'stewartia_pseudocamellia', 'cerasus_vulgaris', 'juglans_regia', 'sorbus_aucuparia', 'platanus_acerifolia', 'hippophae_rhamnoides', 'quercus_robur']
    #arch=resnet34
    data = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(models.resnet34, data)
    learn.load('stage-3')
    return learn

app = Flask(__name__)
app.config['DEBUG'] = False
trained_model = load_model()

@app.route('/', methods=['GET'])
def index():
    """Render the app"""
    return render_template('serving_template.html')

@app.route('/image', methods=['POST'])
def eval_image():
    """Evaluate the image!"""
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File is not present in the request")
    if input_file.filename == '':
        return BadRequest("Filename is not present in the request")
    if not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return BadRequest("Invalid file type")
    
    input_buffer = BytesIO()
    input_file.save(input_buffer)
    
    guess = evaluate_image(open_image(input_buffer))
    return jsonify({
        'guess': guess
    })

if __name__ == "__main__":
    app.run()
