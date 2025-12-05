
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PVlistWidget import PVlistWidget, TableData
import json
import time

from os import path

import logging

logger = logging.getLogger(__name__)

class PVMainWindow(QtWidgets.QWidget):

    pvlistWidget = None

    def __init__(self, parent=None):
        super(PVMainWindow, self).__init__(parent)
        
        self.ui = uic.loadUi(path.join(path.dirname(path.realpath(__file__)), 'opt.ui'), self)
        self.setWindowTitle('PV Table')


        self.all_pv = []

        with open(path.join(path.dirname(path.realpath(__file__)), 'test_pv.txt'), 'r') as f:
            
            self.all_pv = [line[:-1]+':SETI' for line in f]
            
        self.all_pv += [f'SBP-UD:IVU{i:02d}:UN_Gap_Setting' for i in range(1, 11)]
        self.all_pv += ['LA-CN:MOD_16:WRITE_V', 'PIL:SMC2:pa_a1']


        self.table = TableData(self.ui.pv_table)
        # self.ui.pv_table.resizeColumnsToContents()
        self.ui.pv_table.setColumnWidth(0, 300)
        self.ui.pv_table.setColumnWidth(1, 100)
        self.ui.pv_table.setColumnWidth(2, 100)
        self.ui.pv_table.setColumnWidth(3, 100)
        
        
        ###set signals

        self.ui.bound_scale_SpinBox.valueChanged.connect(self._on_bound_scale_changed)


        self.ui.add_pv_from_list_pushButton.clicked.connect(self._on_add_pv_from_list_clicked)

        self.ui.clear_pv_pushButton.clicked.connect(lambda :self.table.set_pvlist([]))
        
        self.ui.save_pushButton.clicked.connect(self._on_save_clicked)


    def ui_filepath(self):
        # Return the full path to the UI file
        from os import path
        return path.join(path.dirname(path.realpath(__file__)), self.ui_filename())


    def _on_bound_scale_changed(self, value: float) -> None:
        self._bound_scale = value
        self.table._bound_scale = value

        self.table.set_pvlist(self.table.get_pvlist())


    def __open_fileDialog(self):
        dialog = QtWidgets.QFileDialog(self)
        dirname = path.join(path.dirname(path.realpath(__file__)), './')
        dialog.setDirectory(dirname)
        # dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Search_space (*.json)")
        dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            return filenames

    def _on_save_clicked(self):

        filenames = self.__open_fileDialog()
        if filenames:
            self.__save_action(filenames[0])


    def __save_action(self, filename=None) -> None:
              
        pvlist, x0, bounds = self.table.get_data()


        data = {pv: {"_type": "uniform", "_value":[v[0], v[1]]} for pv, v in zip(pvlist, bounds)}


            
        if '.json' not in filename:
            filename = filename+'.json'

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

        logger.info(f'Search space saved: {filename}')
 


    def _on_add_pv_from_list_clicked(self) -> None:
        if self.pvlistWidget is None:
            self.pvlistWidget = PVlistWidget(self.all_pv, parent=self)
            self.pvlistWidget.send_selected_items.connect(self.table._on_pvlist_change)

        self.pvlistWidget.show()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mianWindow = PVMainWindow()
    mianWindow.show()
    sys.exit(app.exec_())

