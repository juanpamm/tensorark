from django.shortcuts import render


def upload_text_nn_template(request):
    return render(request, 'textproc/upload_text_set_nn.html')
