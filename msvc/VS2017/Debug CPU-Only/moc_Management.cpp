/****************************************************************************
** Meta object code from reading C++ file 'Management.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.13.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../autogtp/Management.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Management.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.13.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_Management_t {
    QByteArrayData data[11];
    char stringdata0[78];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Management_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Management_t qt_meta_stringdata_Management = {
    {
QT_MOC_LITERAL(0, 0, 10), // "Management"
QT_MOC_LITERAL(1, 11, 8), // "sendQuit"
QT_MOC_LITERAL(2, 20, 0), // ""
QT_MOC_LITERAL(3, 21, 9), // "getResult"
QT_MOC_LITERAL(4, 31, 5), // "Order"
QT_MOC_LITERAL(5, 37, 3), // "ord"
QT_MOC_LITERAL(6, 41, 6), // "Result"
QT_MOC_LITERAL(7, 48, 3), // "res"
QT_MOC_LITERAL(8, 52, 5), // "index"
QT_MOC_LITERAL(9, 58, 8), // "duration"
QT_MOC_LITERAL(10, 67, 10) // "storeGames"

    },
    "Management\0sendQuit\0\0getResult\0Order\0"
    "ord\0Result\0res\0index\0duration\0storeGames"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Management[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   29,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    4,   30,    2, 0x0a /* Public */,
      10,    0,   39,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 4, 0x80000000 | 6, QMetaType::Int, QMetaType::Int,    5,    7,    8,    9,
    QMetaType::Void,

       0        // eod
};

void Management::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<Management *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->sendQuit(); break;
        case 1: _t->getResult((*reinterpret_cast< Order(*)>(_a[1])),(*reinterpret_cast< Result(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3])),(*reinterpret_cast< int(*)>(_a[4]))); break;
        case 2: _t->storeGames(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (Management::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Management::sendQuit)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject Management::staticMetaObject = { {
    &QObject::staticMetaObject,
    qt_meta_stringdata_Management.data,
    qt_meta_data_Management,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *Management::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Management::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_Management.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int Management::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 3)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 3;
    }
    return _id;
}

// SIGNAL 0
void Management::sendQuit()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
