# Generated by Django 3.1.8 on 2022-07-19 22:00

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('recognition', '0003_criminal_profile_profile_pic'),
    ]

    operations = [
        migrations.AlterField(
            model_name='criminallog',
            name='alias_name',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='recognition.criminal_profile'),
        ),
    ]
