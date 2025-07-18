CLASS z_humaneval_047 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: median
      IMPORTING
        !l            TYPE int4_table
      RETURNING
        VALUE(result) TYPE f.

ENDCLASS.

CLASS z_humaneval_047 IMPLEMENTATION.

  METHOD median.
    DATA: lt_sorted TYPE int4_table,
          lv_len    TYPE i,
          lv_mid    TYPE i.

    lt_sorted = l.
    SORT lt_sorted.

    lv_len = lines( lt_sorted ).

    IF lv_len MOD 2 = 1.
      lv_mid = ( lv_len + 1 ) / 2.
      READ TABLE lt_sorted INDEX lv_mid INTO DATA(lv_value).
      result = lv_value.
    ELSE.
      lv_mid = lv_len / 2.
      READ TABLE lt_sorted INDEX lv_mid INTO DATA(lv_value1).
      READ TABLE lt_sorted INDEX ( lv_mid + 1 ) INTO DATA(lv_value2).
      result = ( lv_value1 + lv_value2 ) / 2.
    ENDIF.
  ENDMETHOD.

ENDCLASS.