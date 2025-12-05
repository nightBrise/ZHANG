from PyQt5 import QtCore, QtGui, QtWidgets


from epics import caget_many


from collections import OrderedDict

import logging

logger = logging.getLogger(__name__)

class TableData():

    def __init__(self, table_widget) -> None:
        self.table = table_widget

        self._bound_scale = 0.1

    def get_pvlist(self):
        rownum = self.table.rowCount()

        pvlist = []
        for r in range(rownum):
            pv_item = self.table.item(r, 0)
            pvlist.append(pv_item.text())

        return pvlist

    def get_data(self):
        rownum = self.table.rowCount()

        pvlist = []
        values = []
        bounds = []
        for r in range(rownum):
            pv  = self.table.item(r, 0).text()
            v= float(self.table.item(r, 1).text())
            l = float(self.table.item(r, 2).text())
            u = float(self.table.item(r, 3).text())
            pvlist.append(pv)
            values.append(v)
            bounds.append((l, u))

        return pvlist, values, bounds

    def add_pvlist(self, pvlist):
        if len(pvlist) == 0:
            return
        
        if self.table.rowCount() > 0:
            current_pvlist = self.get_pvlist()
            pvlist = current_pvlist + pvlist
        
        self.set_pvlist(pvlist)

    def set_pvlist(self, pvlist):

        rownum = len(pvlist)
        self.table.setRowCount(rownum)
        if rownum == 0: return

        pvlist = list(OrderedDict.fromkeys(pvlist))
        rownum = len(pvlist)

        

        # if _LOG.level != logging.DEBUG:
        values = caget_many(pvlist, timeout=0.1)
        if None in values:
            # print(values)
            logger.info("PV value initialization failed", exc_info=True)
            logger.info(values)
            values = values = caget_many(pvlist, timeout=0.2)
            if None in values:
                values = [0.0]*len(pvlist)

        bounds = create_relative_bounds(pvlist, self._bound_scale)

        for r, pv, value, bound in zip(range(rownum), pvlist, values, bounds):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(pv))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(f'{value:.4}'))
            lower = value + bound[0]
            upper = value + bound[1]
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(f'{lower:.4}'))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(f'{upper:.4}'))

    # @QtCore.pyqtSlot(list)
    def _on_pvlist_change(self, pvlist):
        self.add_pvlist(pvlist)
    
def create_relative_bounds(pvs, scale=1.):
    bounds = []
    for ipv in pvs:
        ibounds = (-0.2*scale, 0.2*scale)
        if 'Q' in ipv:
            ibounds = (-0.2*scale, 0.2*scale)
        elif 'PIL:SMC2:pa_a1' in ipv:
            ibounds = (-0.2*scale, 0.2*scale)
        elif 'IN-PS:S01' in ipv:
            ibounds = (-0.5*scale, 0.5*scale)
        elif 'WRITE_V' in ipv:
            ibounds = (-0.4*scale, 0.4*scale)
        elif 'SET_PHASE' in ipv:
            ibounds = (-2*scale, 2*scale)
        elif 'SET_AMP' in ipv:
            ibounds = (-0.5*scale, 0.5*scale)
        elif ('UND' in ipv or 'IVU' in ipv) and 'GAP' in ipv.upper() and 'SET' in ipv.upper():
            ibounds = (-0.05*scale, 0.05*scale)
        elif 'FC' in ipv:
            ibounds = (-3*scale, 3*scale)

        bounds.append(ibounds)
    return bounds

class PVlistWidget(QtWidgets.QDialog):
    send_selected_items = QtCore.pyqtSignal(list)

    def __init__(self, list_data, parent=None):
        super(PVlistWidget, self).__init__(parent)

        self.all_list_data = list_data
        
        self.setMinimumSize(150, 1000)

        self.le = QtWidgets.QLineEdit('', textChanged=self.on_textChanged)
        self.lv = QtWidgets.QListView()

        self.add_btn = QtWidgets.QPushButton('Add selected pv')

        self.model = QtGui.QStandardItemModel(self.lv)

        for ld in list_data:
            item = QtGui.QStandardItem(ld)
            # item.setCheckable(True)
            self.model.appendRow(item)
            
        self.proxy_model = QtCore.QSortFilterProxyModel(
            recursiveFilteringEnabled=True)
        self.proxy_model.setSourceModel(self.model)

        self.lv.setModel(self.proxy_model)
        # self.adjust_root_index()

        # self.lv.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.lv.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.le)
        lay.addWidget(self.lv)
        lay.addWidget(self.add_btn)

        # self.send_selected_items = QtCore.pyqtSignal(list)
        
        self.add_btn.clicked.connect(self.on_add_selected)

    @QtCore.pyqtSlot(str)
    def on_textChanged(self, text):
        self.proxy_model.setFilterWildcard("*{}*".format(text))

    @QtCore.pyqtSlot()
    def on_add_selected(self):
        _text = self.le.text()

        self.le.setText('')
        ind_list = [i.row() for i in self.lv.selectedIndexes()]
        ind_list = sorted(ind_list)
    
        self.le.setText(_text)

        data = [self.all_list_data[i] for i in ind_list]

        self.send_selected_items.emit(data)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    list_data = [f'{i}1' for i in range(100)]
    w = PVlistWidget(list_data)
    w.show()
    sys.exit(app.exec_())
