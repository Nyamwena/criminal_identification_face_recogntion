# Generated by Django 3.1.8 on 2022-06-28 20:32

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Criminal_Profile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('alias_name', models.CharField(max_length=30, unique=True)),
                ('firstname', models.CharField(max_length=30)),
                ('lastname', models.CharField(max_length=30)),
                ('national_id_number', models.CharField(max_length=20, unique=True)),
                ('dob', models.DateField()),
                ('nationality', models.CharField(max_length=30)),
                ('place_of_orgin', models.CharField(max_length=30)),
                ('address', models.CharField(max_length=30)),
                ('phone_number', models.CharField(max_length=30)),
                ('longitude', models.CharField(blank=True, max_length=200, null=True)),
                ('latitude', models.CharField(blank=True, max_length=30, null=True)),
                ('bio', models.CharField(blank=True, max_length=200)),
                ('crimes_comitted', models.CharField(blank=True, max_length=200, null=True)),
                ('date', models.DateField(default=datetime.date.today)),
            ],
        ),
    ]
