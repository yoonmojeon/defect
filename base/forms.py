# base/forms.py

from django import forms
from .models import Image

class UploadForm(forms.ModelForm):
    text = forms.CharField(widget=forms.Textarea)
    class Meta:
        model = Image
        fields = ('image',)
        
        
class EditTextForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea)
