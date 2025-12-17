from django import forms

class LoginForm(forms.Form):
    user_id = forms.CharField(
        label="아이디",
        max_length=20,
        widget=forms.TextInput(attrs={
            "class": "form-input",
        })
    )
    
    password = forms.CharField(
        label="비밀번호",
        widget=forms.PasswordInput(attrs={
            "class": "form-input",
        })
    )
