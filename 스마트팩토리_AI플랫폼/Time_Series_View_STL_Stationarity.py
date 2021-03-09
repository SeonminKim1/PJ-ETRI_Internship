"""
	Knowledge-based AI Framework for Smart Factory - SW-SoC Open Platform Research Section")
	#
	# file        : gui_pm.py
	#
	# Description : Predict outputs base on input columns using Regressions or Classifications
	# Author      : Wonjong Kim
	# Brand       : ETRI
	# E-mail      : wjkim@etri.re.kr
	# Website     : www.etri.re.kr
	# Ver. 1.0    : 2019-06-08,	Wonjong Kim
	#	Added Feature Extraction
	# Ver. 1.1    : 2019-07-29,	Wonjong Kim
	#	Changed PyQt Version to 5
	# Ver. 1.2    : 2019-08-28, Wonjong Kim
	#	Added Classifications,		
	# Ver. 1.2.1  : 2019-08-29,	Wonjong Kim
	#	Added Scaling Selection for Classifiers
	# Ver. 1.3    : 2019-09-02,	Wonjong Kim
	#	Added AutoML(TPOT) for Hyper Parameter Optimization
	#
	# Ver. 1.4    : 2021-01-29,	Seonmin Kim
	#	Added TimeSeries View & Decomposition mode
	# Ver. 1.5    : 2021-02-08,	Seonmin Kim
	#	Added Stationarity mode
	# Ver. 1.5.1    : 2021-02-09,	Seonmin Kim
	#	Added Statinoarity Metrics - ADF Test
	# Ver. 1.6    : 2021-02-26      Seonmin Kim
	#	Merged Stationarity & STL Mode
	# 
	# Requirements: QtDesigner, PyQt4, scipy, sklearn, xgboost, matplotlib, tpot, statsmodels
"""

#-*- coding:utf-8 -*-

from config_gui_pm import *		# Configuration of the GUI, PyQt Ver. 4 or 5

if PYQT_VER == 5:
	from PyQt5 import *
	from PyQt5.QtCore import *
	from PyQt5.QtGui import *
	from PyQt5.QtWidgets import *
	from MenuBar_Grid import *
elif PYQT_VER == 4:
	from PyQt4 import *
	#from PyQt4 import QtGui, QtCore
	from PyQt4.QtCore import *
	from PyQt4.QtGui import *
	from MenuBar_Grid5 import *
else:
	print("Unexpected PYQT_VER: ", PYQT_VER)
	exit(0)

#from MenuBar_Grid import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pandas as pd
import datetime
from dateutil import parser
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

#TS_mode_Types = ['Stationarity', 'STL']
Period_Mode_Types = ['milliseconds','seconds','minutes','hours','days','months','years']
Aggregation_Mode_Types = ['Mean', 'Sum', 'Median','Max','Min']

try:
	_fromUtf8 = QtCore.QString.fromUtf8

except AttributeError:
	print("QtCore.QString.fromUtf8 exception in gui_pm.py")
	def _fromUtf8(s):
		return s

print("PyQt Version: ", PYQT_VER)

if PYQT_VER == 5:
	print("use str() instead of QString() for PyQt5")
	def	QString():
		return str()

try:
	_encoding = QApplication.UnicodeUTF8
	def _translate(context, text, disambig):
		return QApplication.translate(context, text, disambig, _encoding)

except AttributeError:
	def _translate(context, text, disambig):
		return QApplication.translate(context, text, disambig)

class TS_Data_View_Dialog(QDialog):
	# Diaglog to View/Plot Columns Data
	# make basic window
	def __init__(self, parent):
		super(TS_Data_View_Dialog, self).__init__()

		# list parameter
		self.available_date_column=[]
		self.available_target_column = []
		self.checkbox_date = []
		self.checkbox_target = []

		# input column + candidates columns
		for col in parent.df.columns:
			if self.is_date(parent.df[col].loc[0]):
				self.available_date_column.append(col)
				if col in parent.inputs:
					self.checkbox_date.append(col)

		diff_target_columns = (parent.df.columns).difference(self.available_target_column)

		# output column + candidates columns
		for col in diff_target_columns:
			if np.issubdtype(parent.df[col].dtype, np.number):
				self.available_target_column.append(col)
				if col in parent.outputs:
					self.checkbox_target.append(col)

		print('All column', parent.df.columns)
		print('Input column', parent.inputs, '\nOutput column', parent.outputs)
		print('Final input column', self.available_date_column, 'checkbox_date', self.checkbox_date)
		print('Final output column', self.available_target_column, 'checkbox_target', self.checkbox_target)

		# validation check - not timeseries data!
		if len(self.available_date_column)==0:
			print('Not exist any timeseries data columns')
			msg_box=QMessageBox()
			msg_box.setIcon(QMessageBox.Critical)
			msg_box.question(self, 'Input Data Type Error', 'Not exist any timeseries column data',QMessageBox.Ok)
			return

		if len(self.available_target_column) == 0:
			print('Not exist any numeric data columns')
			msg_box=QMessageBox()
			msg_box.setIcon(QMessageBox.Critical)
			msg_box.question(self, 'Output Data Type Error', 'Not exist any numeirc column data ',QMessageBox.Ok)
			return

		# date column valid check
		print('Check_ts_input_data...  OK !!')
		print('x column - ', self.available_date_column)
		print('y column - ', self.available_target_column)


		# main form create
		self.createFormGroupBox(parent)

		buttonBox = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Ok)
		buttonBox.button(QDialogButtonBox.Apply).clicked.connect(lambda: self.Apply(parent))
		buttonBox.accepted.connect(self.accept)

		mainLayout = QVBoxLayout()
		HLayout = QHBoxLayout()

		scrollarea = QScrollArea(self)
		scrollarea.setMinimumWidth(100)
		scrollarea.setMinimumHeight(200)
		scrollarea.setWidgetResizable(True)
		scrollarea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
		scrollarea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
		scrollarea.setFixedSize(220, 800)
		scrollarea.setWidget(self.formGroupBox)

		HLayout.addWidget(scrollarea)

		self.plot_widget = matplotlibWidget()  # Data Plot
		HLayout.addWidget(self.plot_widget)
		mainLayout.addLayout(HLayout)
		mainLayout.addWidget(buttonBox)

		self.setLayout(mainLayout)
		self.resize(1500, 800)

		self.setWindowTitle("Select Columns")

	# date type check -> data[col] is date?
	def is_date(self, obj):
		if  isinstance(obj, datetime.date):
			return True
		else:
			try:
				# iso8601 - data elements and interchange formats
				parser.isoparse(obj)
	#			print(parser.isoparse(obj))
				return True
			except:
				return False

################# - View Data ###################################
	def createFormGroupBox(self, parent):  # Show column names to select
		# Check boxes for Column selection
		self.formGroupBox = QGroupBox("Time Series Analyze")

		layout = QFormLayout()

		# x axis column
		layout.addRow(QLabel("==   Column View   =="))
		layout.addRow(QLabel("                     "))
		layout.addRow(QLabel("==   X axis list   =="))

		cbs_x=[];
		for col in self.available_date_column:
			cb = QCheckBox()
			if len(self.available_date_column) !=1:
				cb.setEnabled(False)

			# set column menu - input checked sign
			if len(self.checkbox_date)!=0:
				if col in self.checkbox_date[0]: # 1 xaxis
					cb.setChecked(True)
					cb.setEnabled(True)

			cbs_x.append(cb)
			layout.addRow(QLabel(col), cb)
			cb.stateChanged.connect(self.checkChanged_x)
		self.cbs_x = cbs_x

		# y axis list draw
		layout.addRow(QLabel("                     "))
		layout.addRow(QLabel("==   Y axis list   =="))

		self.num_y_list = []
		cbs_y=[];
		# y axis column
		for col in self.available_target_column:
			cb = QCheckBox()
			layout.addRow(QLabel(col), cb)
			cbs_y.append(cb)
			self.num_y_list.append(col) # numeric column

			# set column menu - input checked sign
			if col in self.checkbox_target:
				cb.setChecked(True)
		self.cbs_y = cbs_y

		# Time Series Setting
		layout.addRow(QLabel("                         "))
		layout.addRow(QLabel("== Time Series Setting =="))

		# period combobox, aggregation combobox
		period_cbs, agg_cbs = QComboBox(), QComboBox()
		t_edit = QLineEdit()
		t_edit.setText('1')
		t_edit.setValidator(QIntValidator())
		self.time_edit = t_edit

		# period_mode_types
		for mode_type in Period_Mode_Types:
			period_cbs.addItem(mode_type)
		self.period_mode_cbs = period_cbs

		default_time_unit_number = self.get_default_time_unit(parent)
		self.period_mode_cbs.setCurrentIndex(default_time_unit_number)

		# aggregation
		for agg_type in Aggregation_Mode_Types:
			agg_cbs.addItem(agg_type)
		self.agg_mode_cbs = agg_cbs

		layout.addRow(QLabel("Time Setting"), self.time_edit)
		layout.addRow(QLabel("Time Unit"), self.period_mode_cbs)
		layout.addRow(QLabel("Aggregation"), self.agg_mode_cbs)

		self.formGroupBox.setLayout(layout)

	def get_default_time_unit(self, parent):
		if len(self.checkbox_date)!=0:
			k = pd.to_datetime(parent.df[self.checkbox_date[0]]).sort_values().drop_duplicates()
		else:
			k = pd.to_datetime(parent.df[self.available_date_column[0]]).sort_values().drop_duplicates()

		print(k.iloc[1], k.iloc[0])
		time_range = (k.iloc[1]) - (k.iloc[0])
		# number is 'Period_Mode_Types' list
		# ex)Period_Mode_Types = ['milliseconds','seconds','minutes','hours','days','months','years']
		print('default_time_unit', time_range)

		if time_range.days>=28:	return 5 # months, years
		elif time_range.days>=1: return 4 # d
		elif time_range.hours>=1: return 3 # h
		elif time_range.minutes>=1: return 2 # m
		elif time_range.seconds>=1: return 1 # s
		elif time_range.milliseconds>=1: return 0 # ms
		else : return 4 # etc, default days

	def checkChanged_x(self):
		for i in self.cbs_x:
			i.setEnabled(False)

		flag=False
		for i in self.cbs_x:
			if i.isChecked():
				i.setEnabled(True)
				flag=True
		if flag==False:
			for i in self.cbs_x:
				i.setEnabled(True)
			print('checkcnahged_x')

	def Apply(self, parent):  # Draw selected columns
		# First find selected columns
		from matplotlib.figure import Figure

		##### x axis y axis Check
		# ts x axis, y axis
		selected_x, selected_y = [], []

		# self.available_date_column => x axis
		# self.feature_column_list => y axis
		idx = 0 # check column num
		for i, cb in enumerate(self.cbs_x): # checkbox x axis
			checked = cb.isChecked() #
			if checked == 1:
				selected_x.append(self.available_date_column[i])

		for j, cb in enumerate(self.cbs_y): # checkbox y axis
			checked = cb.isChecked()
			if checked == 1:
				selected_y.append(self.num_y_list[j])
		print('x axis', selected_x, 'y axis', selected_y)

		# x data check
		if len(selected_x)<=0:
			print("There is no X data selected!")
			return

		else: # C
			print('X Data Selection Completed')

		# y data check
		if len(selected_y)<=0:
			print("There is no Y data selected!")
			return

		elif len(selected_y)==1:
			use_subplots=False
			print('Selection y completed', selected_y)
		else: # Multi y
			use_subplots=True
			print('Selection y completed', selected_y)

		data = parent.df.copy()
		# For TS Analayze, convert timeseries to datetime type
		if isinstance(data[selected_x[0]].loc[0], datetime.date)==False:
			data[selected_x[0]] = pd.to_datetime(data[selected_x[0]])

		# --- parameter clear & period setting start --- #
		print('ts parameter checking...')
		data = data.set_index(selected_x[0])[selected_y]

		##### TimeSeries Setting Check
		# time input, time unit combo box, aggreagte combo box
		time_text = str(self.time_edit.text())
		pcb_text = str(self.period_mode_cbs.currentText());
		acb_text = str(self.agg_mode_cbs.currentText());

		# time_edit null check
		if time_text == "" or time_text == '0':
			print('Please refill in the Time blank ')
			return

		print('time_text', time_text, 'pcb_text', pcb_text, 'acb_text', acb_text)
		result_data = self.resample_data(data, time_text, pcb_text, acb_text) # resample data ouput
		print(result_data.head(), type(result_data))

		##### 5. Graph Visualization
		print(result_data.head())
		date_index = result_data.index
		print('date_index\n', type(date_index), date_index)
		#date_index_len = len(date_index)
		#print('date_index_len', date_index_len, int(date_index_len/10))
		#print(date_index[::int(date_index_len/10)])

		self.plot_widget.canvas.ax.clear()

		# single graph
		if len(selected_y) == 1:
#			del self.plot_widget.canvas.ax  # remove and recreate axes to solve problem after using subplot
			self.plot_widget.canvas.ax = self.plot_widget.canvas.fig.add_subplot(111)
			self.plot_widget.canvas.ax.clear()
			self.plot_widget.canvas.ax.plot(result_data[selected_y])
#			self.plot_widget.canvas.ax.set_xticks(date_index[::date_index_len])

		# multi graph
		else: # A
#			self.plot_widget.canvas.ax.plot(result_data[selected_y])
		#	self.plot_widget.canvas.ax.set_xticks(result_data.index[::int(date_index_len/10)])
		#	self.plot_widget.canvas.fig.gca()
			result_data[selected_y].plot(ax=self.plot_widget.canvas.ax, subplots=use_subplots)

		self.plot_widget.canvas.draw()
		self.plot_widget.canvas.show()

	# resample
	def resample_data(self, data, time_text, pcb_text, acb_text):
		if pcb_text =='milliseconds':
			result = self.resample_acb_data(data, time_text+'ms', acb_text)
		elif pcb_text =='seconds':
			result = self.resample_acb_data(data, time_text+'s', acb_text)
		elif pcb_text =='minutes':
			result = self.resample_acb_data(data, time_text+'t', acb_text)
		elif pcb_text =='hours':
			result = self.resample_acb_data(data, time_text+'h', acb_text)
		elif pcb_text =='days':
			result = self.resample_acb_data(data, time_text+'d', acb_text)
		elif pcb_text =='months':
			result = self.resample_acb_data(data, time_text+'m', acb_text)
		elif pcb_text =='years':
			result = self.resample_acb_data(data, time_text+'y', acb_text)
		return result


	def resample_acb_data(self, data, frequency, acb_text):
		if acb_text == 'Sum':return data.resample(frequency).sum().fillna(0)
		elif acb_text =='Mean':	return data.resample(frequency).mean().fillna(0)
		elif acb_text =='Median':return data.resample(frequency).median().fillna(0)
		elif acb_text =='Max':return data.resample(frequency).max().fillna(0)
		elif acb_text =='Min':return data.resample(frequency).min().fillna(0)

##################################################################################

########### STL_Stationarity
class TS_STL_Stationarity_Dialog(QDialog):
	# make basic window
	def __init__(self, parent):
		super(TS_STL_Stationarity_Dialog, self).__init__()

		self.after_stationarity_data = None
		self.after_stationarity_flag = False

		# input, output parameter list for timeseries
		self.available_date_column=[]
		self.available_target_column = []
		self.checkbox_date = []
		self.checkbox_target = []

		# input column + candidates columns
		for col in parent.df.columns:
			if self.is_date(parent.df[col].loc[0]):
				self.available_date_column.append(col)
				if col in parent.inputs:
					self.checkbox_date.append(col)

		diff_target_columns = (parent.df.columns).difference(self.available_target_column)

		# output column + candidates columns
		for col in diff_target_columns:
			if np.issubdtype(parent.df[col].dtype, np.number):
				self.available_target_column.append(col)
				if col in parent.outputs:
					self.checkbox_target.append(col)

		print('All column', parent.df.columns)
		print('Input column', parent.inputs, '\nOutput column', parent.outputs)
		print('Final input column', self.available_date_column, 'checkbox_date', self.checkbox_date)
		print('Final output column', self.available_target_column, 'checkbox_target', self.checkbox_target)

		# validation check - not timeseries data!
		if len(self.available_date_column)==0:
			print('Not exist any timeseries data columns')
			msg_box=QMessageBox()
			msg_box.setIcon(QMessageBox.Critical)
			msg_box.question(self, 'Input Data Type Error', 'Not exist any timeseries column data',QMessageBox.Ok)
			return

		if len(self.available_target_column) == 0:
			print('Not exist any numeric data columns')
			msg_box=QMessageBox()
			msg_box.setIcon(QMessageBox.Critical)
			msg_box.question(self, 'Output Data Type Error', 'Not exist any numeirc column data ',QMessageBox.Ok)
			return

		# date column valid check
		print('Check_ts_input_data...  OK !!')
		print('x column - ', self.available_date_column)
		print('y column - ', self.available_target_column)

		# main form create
		self.createFormGroupBox(parent)

		buttonBox = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Ok)
		buttonBox.button(QDialogButtonBox.Apply).clicked.connect(lambda: self.Apply(parent))
		buttonBox.accepted.connect(self.accept)

		mainLayout = QVBoxLayout()
		HLayout = QHBoxLayout()

		scrollarea = QScrollArea(self)
		scrollarea.setMinimumWidth(100)
		scrollarea.setMinimumHeight(200)
		scrollarea.setWidgetResizable(True)
		scrollarea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
		scrollarea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
		scrollarea.setFixedSize(220, 800)
		scrollarea.setWidget(self.formGroupBox)

		HLayout.addWidget(scrollarea)

		self.plot_widget = matplotlibWidget()  # Data Plot
		HLayout.addWidget(self.plot_widget)
		mainLayout.addLayout(HLayout)
		mainLayout.addWidget(buttonBox)

		self.setLayout(mainLayout)
		self.resize(1500, 800)

		self.setWindowTitle("Select Columns")

	# date type check -> data[col] is date?
	def is_date(self, obj):
		if  isinstance(obj, datetime.date):
			return True
		else:
			try:
				# iso8601 - data elements and interchange formats
				parser.isoparse(obj)
	#			print(parser.isoparse(obj))
				return True
			except:
				return False

################# - Data STL & Stationarity Check - ##############################
	def createFormGroupBox(self, parent):  # Show column names to select
		# Check boxes for Column selection
		self.formGroupBox = QGroupBox("Time Series Analysis")
		layout = QFormLayout()

		# Mode Combobox - Stationarity vs STL Decompose
		layout.addRow(QLabel("==  Stationarity & STL Decomposition =="))
		layout.addRow(QLabel("                                       "))
		layout.addRow(QLabel("==     Columns View     =="))
		layout.addRow(QLabel("                          "))
		layout.addRow(QLabel("==      X axis list     =="))

		########### column list ###############
		# checkbox x list
		cbs_x=[];
		for col in self.available_date_column:
			cb = QCheckBox()
			if len(self.available_date_column) !=1:
				cb.setEnabled(False)

			# set column menu - input checked sign
			if len(self.checkbox_date)!=0:
				if col in self.checkbox_date[0]: # 1 xaxis
					cb.setChecked(True)
					cb.setEnabled(True)

			cbs_x.append(cb)
			layout.addRow(QLabel(col), cb)
			cb.stateChanged.connect(self.checkChanged_x)
		self.cbs_x = cbs_x

		# y axis list draw
		layout.addRow(QLabel("                     "))
		layout.addRow(QLabel("==   Y axis list   =="))

		# checkbox y list
		self.num_y_list = [] # numeric y list
		cbs_y=[];
		# y axis column
		for col in self.available_target_column:
			cb = QCheckBox()
			if len(self.available_date_column) !=1:
				cb.setEnabled(False)

			# set column menu
			if len(self.checkbox_target)!=0:
				if col in self.checkbox_target[0]: # 1 yaxis
					cb.setChecked(True)
					cb.setEnabled(True)

			cbs_y.append(cb)
			layout.addRow(QLabel(col), cb)
			self.num_y_list.append(col) # numeric column
			cb.stateChanged.connect(self.checkChanged_y)
		self.cbs_y = cbs_y

		################ Time Series Setting ###################
		layout.addRow(QLabel("                         "))
		layout.addRow(QLabel("== Time Series Setting =="))

		## period combobox, aggregation combobox
		period_cbs, agg_cbs = QComboBox(), QComboBox()
		t_edit = QLineEdit()
		t_edit.setText('1')
		t_edit.setValidator(QIntValidator())
		self.time_edit = t_edit

		### period_mode_types
		for mode_type in Period_Mode_Types:
			period_cbs.addItem(mode_type)
		self.period_mode_cbs = period_cbs

		default_time_unit_number = self.get_default_time_unit(parent)
		self.period_mode_cbs.setCurrentIndex(default_time_unit_number)

		### aggregation
		for agg_type in Aggregation_Mode_Types:
			agg_cbs.addItem(agg_type)
		self.agg_mode_cbs = agg_cbs

		layout.addRow(QLabel("Time Setting"), self.time_edit)
		layout.addRow(QLabel("Time Unit"), self.period_mode_cbs)
		layout.addRow(QLabel("Aggregation"), self.agg_mode_cbs)


		## STL & Stationarity Setting
		layout.addRow(QLabel("                   "))
		layout.addRow(QLabel("==  STL Setting  =="))

		## STL - radio button
		self.stl_radio_group = QGroupBox('STL')

		self.stl_radio_btn1 = QRadioButton('Additive')
		self.stl_radio_btn1.setChecked(True)
		self.stl_radio_btn1.mode='Additive'

		self.stl_radio_btn2 = QRadioButton('Multiplicative')
		self.stl_radio_btn2.mode='Multiplicative'

		r_vbox = QVBoxLayout()
		self.stl_radio_group.setLayout(r_vbox)

		r_vbox.addWidget(self.stl_radio_btn1)
		r_vbox.addWidget(self.stl_radio_btn2)

#		self.stl_radio_group.addButton(self.stl_radio_btn1)
#		self.stl_radio_group.addButton(self.stl_radio_btn2)

		layout.addRow(self.stl_radio_group)

		#hbox.addWidget(self.stl_radio_group); #hbox.addWidget(self.stl_radio_btn2)
		#layout.addRow(QLabel("Decomposition"), hbox)
		layout.addRow(QLabel("                            "))
		layout.addRow(QLabel("==  Stationarity Setting  =="))

		## Stationarity - diff time edit
		t_diff_edit = QLineEdit()
		t_diff_edit.setText('1')
		t_diff_edit.setValidator(QIntValidator())
		self.time_diff_edit = t_diff_edit
		layout.addRow(QLabel("Diff time"), self.time_diff_edit)

		## Stationarity - log checkbox Button
		self.log_cb = QCheckBox()
		self.log_cb.setEnabled(True) # initalize
		layout.addRow(QLabel("Log tramsform"), self.log_cb)

		### Stationarity - Diff - radio

		## STL - radio button
		self.diff_radio_group = QGroupBox('Difference')

		self.diff_radio_btn1 = QRadioButton('1st')
		self.diff_radio_btn1.setChecked(True)
		self.diff_radio_btn1.mode='1st'

		self.diff_radio_btn2 = QRadioButton('2st (contain 1st)')
		self.diff_radio_btn2.mode='2st'

		r_vbox2 = QVBoxLayout()
		self.diff_radio_group.setLayout(r_vbox2)

		r_vbox2.addWidget(self.diff_radio_btn1)
		r_vbox2.addWidget(self.diff_radio_btn2)

		layout.addRow(self.diff_radio_group)

		## ADF Metircs
		self.adf_statistics_lb = QLabel("0")
		self.adf_pvalue_lb = QLabel("0")
		self.adf_result =QLabel("None")

		layout.addRow(QLabel("ADF Statistics:"), self.adf_statistics_lb)
		layout.addRow(QLabel("ADF p-value:"), self.adf_pvalue_lb)
		layout.addRow(QLabel("ADF Result:"), self.adf_result)

		# checkbox changed
		self.formGroupBox.setLayout(layout)

	def get_default_time_unit(self, parent):
		if len(self.checkbox_date)!=0:
			k = pd.to_datetime(parent.df[self.checkbox_date[0]]).sort_values().drop_duplicates()
		else:
			k = pd.to_datetime(parent.df[self.available_date_column[0]]).sort_values().drop_duplicates()

		print(k.iloc[1], k.iloc[0])
		time_range = (k.iloc[1]) - (k.iloc[0])
		# number is 'Period_Mode_Types' list
		# ex)Period_Mode_Types = ['milliseconds','seconds','minutes','hours','days','months','years']
		print('default_time_unit', time_range)

		if time_range.days>=28:	return 5 # months, years
		elif time_range.days>=1: return 4 # d
		elif time_range.hours>=1: return 3 # h
		elif time_range.minutes>=1: return 2 # m
		elif time_range.seconds>=1: return 1 # s
		elif time_range.milliseconds>=1: return 0 # ms
		else : return 4 # etc, default days

	# unchecked, checkbox initalizing
	def set_unchecked_xy(self):
		for i in self.cbs_x:
			i.setChecked(False)
			i.setEnabled(True)

		for  i in self.cbs_y:
			i.setChecked(False)
			i.setEnabled(True)

	# unchecking x axis columns
	def checkChanged_x(self):
		for i in self.cbs_x:
			i.setEnabled(False)

		flag=False
		for i in self.cbs_x:
			if i.isChecked():
				i.setEnabled(True)
				flag=True
		if flag==False:
			for i in self.cbs_x:
				i.setEnabled(True)
			print('checkcnahged_x')

	# unchecking y axis columns
	def checkChanged_y(self):
		#if self.ts_mode_type_cbs.currentText() == 'Decomposition':
		for i in self.cbs_y:
			i.setEnabled(False)

		flag=False
		for i in self.cbs_y:
			if i.isChecked():
				i.setEnabled(True)
				flag=True
		if flag==False:
			for i in self.cbs_y:
				i.setEnabled(True)
			print('checkcnahged_y')

	# Stationarity : 1,2,3,4,5,6,7,9,10
	# STL(Decomposition) : 1,2,3,4,5,6,7,8,10
	def Apply(self, parent):  # Draw selected columns
		# First find selected columns
		# self.available_date_column => all x axis
		# self.available_target_column => all y axis

		from matplotlib.figure import Figure

		##### 1. x axis y axis Check == make selected_x,y
		# ts x axis, y axis
		selected_x, selected_y = [], []

		idx = 0 # check column num
		for i, cb in enumerate(self.cbs_x): # checkbox x axis
			checked = cb.isChecked() #
			if checked == 1:
				selected_x.append(self.available_date_column[i])

		for j, cb in enumerate(self.cbs_y): # checkbox y axis
			checked = cb.isChecked()
			if checked == 1:
				selected_y.append(self.num_y_list[j])
		print('x axis', selected_x, 'y axis', selected_y)

		# 2. X data check
		if len(selected_x)<=0:
			print("There is no selected X data!")
			return

		else:
			print('Selection X Completed', selected_x )

		# 3. y data check
		if len(selected_y)<=0:
			print("There is no selected Y data!")
			return

		elif len(selected_y)==1:
			use_subplots=False
			print('Selection Y Completed', selected_y)
		else: # Multi y
			use_subplots=True
			print('Selection Y Completed', selected_y)

		# 4. For TS Analayze, convert timeseries to datetime type
		data = parent.df.copy()
		if isinstance(data[selected_x[0]].loc[0], datetime.date)==False:
			data[selected_x[0]] = pd.to_datetime(data[selected_x[0]])

		# 5. Timeseries option setting
		print('--TS paramete setting...')
		data = data.set_index(selected_x[0])[selected_y]

		# time input, time unit combo box, aggreagte combo box
		time_text = str(self.time_edit.text())
		pcb_text = str(self.period_mode_cbs.currentText());
		acb_text = str(self.agg_mode_cbs.currentText());

		# time_edit null check
		if time_text == "" or time_text == '0':
			print('Please refill in the Time blank ')
			return

		print('Resampling Option', time_text, pcb_text, acb_text)

		# 6. Data Resampling
		self.result_data = self.resample_data(data, time_text, pcb_text, acb_text) # resample data output
		print('TimeSetting data by Resampling \n', self.result_data.shape, self.result_data.head())
		raw_result_data = self.result_data[selected_y].copy()

##################################################################

		# 7. value validation
		# Time diff text
		time_diff_text = str(self.time_diff_edit.text())
		if time_diff_text =="":
			print('Please refill in the Diff time blank')
			return

		# diff none
		if time_diff_text =='0':
			print('stationarity mode off')
		else:
			# log checkbox diff
			if self.log_cb.isChecked():
				print('log_cb is checked')
				self.result_data[selected_y] = np.log(self.result_data[selected_y])

			# diff - 1st diff
			if self.diff_radio_btn1.isChecked():
				self.result_data[selected_y] = self.result_data[selected_y].diff(periods=int(time_diff_text))
				print('1st diff ok')

			# diff - 2st diff
			elif self.diff_radio_btn2.isChecked():
				self.result_data[selected_y] = self.result_data[selected_y].diff(periods=int(time_diff_text)).diff()
				print('2st diff ok (contain 1st diff)')

		# 8. ADF & KPSS TEST
		print(self.result_data[selected_y].fillna(0).squeeze().values.shape)

		try:
			adf_value = adfuller(self.result_data[selected_y].dropna().squeeze().values)
		except:
			print('data is wrong: datetime is too short')
			return

		print('before adf test shape', self.result_data[selected_y].shape, 'after adf test shape', self.result_data[selected_y].fillna(0).squeeze().shape)
		self.adf_statistics_lb.setText(str(np.round(adf_value[0],4)))

		adf_value = np.round(adf_value[1], 4)
		self.adf_pvalue_lb.setText(str(adf_value))

		if adf_value <= 0.05:
			self.adf_result.setText(' Stationarity is satisfied!')
		else:
			self.adf_result.setText(' Stationarity is unsatisfied!')

		# 9. STL Decomposition Check
		try:
			print('--selected_y', selected_y)
			if self.stl_radio_btn1.isChecked():
				self.decomp = sm.tsa.seasonal_decompose(self.result_data[selected_y].fillna(0), model='additive')
				print('STL Decomposition - additive completed..!!')

			elif self.stl_radio_btn2.isChecked():
				# negative number x / datetime too short x / zeros in the row x
				self.decomp = sm.tsa.seasonal_decompose(self.result_data[selected_y].fillna(0), model='multiplicative')
				print('STL Decomposition - multiplicative completed..!!')

		except:
			print('STL data is wrong: datetime is too short or multiplicative can not work if there are zeros/negative number in the data ')
			return

		# 10. graph visualization
		self.plot_widget.canvas.ax.clear()

		## STL & Decompose
		# raw result data + stl( type 4)
		o1 = raw_result_data.squeeze()
		st1 = self.decomp.observed.squeeze()
		t1 = self.decomp.trend.squeeze()
		s1 = self.decomp.seasonal.squeeze()
		r1 = self.decomp.resid.squeeze()
		# print(type(o1), o1.name, type(t1), t1.name, type(s1), s1.name, type(r1), r1.name)
		#st1 = self.result_data[selected_y].squeeze()

		stationarity_decomp_plot = pd.concat([o1, st1, t1, s1, r1], axis=1)
		stationarity_decomp_plot.columns=['observed', 'stationarity_graph', 'trend', 'seasonal', 'resid']

		# plot view
		stationarity_decomp_plot.fillna(0).plot(ax=self.plot_widget.canvas.ax, subplots=True)
		print('Completed Stationarity & STL Decompose Go to Forecast Menu')

		self.plot_widget.canvas.draw()
		self.plot_widget.canvas.show()

	# resample
	def resample_data(self, data, time_text, pcb_text, acb_text):
		if pcb_text =='milliseconds':
			result = self.resample_acb_data(data, time_text+'ms', acb_text)
		elif pcb_text =='seconds':
			result = self.resample_acb_data(data, time_text+'s', acb_text)
		elif pcb_text =='minutes':
			result = self.resample_acb_data(data, time_text+'t', acb_text)
		elif pcb_text =='hours':
			result = self.resample_acb_data(data, time_text+'h', acb_text)
		elif pcb_text =='days':
			result = self.resample_acb_data(data, time_text+'d', acb_text)
		elif pcb_text =='months':
			result = self.resample_acb_data(data, time_text+'m', acb_text)
		elif pcb_text =='years':
			result = self.resample_acb_data(data, time_text+'y', acb_text)
		return result


	def resample_acb_data(self, data, frequency, acb_text):
		if acb_text == 'Sum':return data.resample(frequency).sum().fillna(0)
		elif acb_text =='Mean':	return data.resample(frequency).mean().fillna(0)
		elif acb_text =='Median':return data.resample(frequency).median().fillna(0)
		elif acb_text =='Max':return data.resample(frequency).max().fillna(0)
		elif acb_text =='Min':return data.resample(frequency).min().fillna(0)

##################################################################################
