CLASS z_humaneval_120 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: maximum
      IMPORTING
        VALUE(arr)    TYPE int4_table
        VALUE(k)      TYPE i
      RETURNING
        VALUE(result) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_120 IMPLEMENTATION.

  METHOD maximum.
    DATA: temp_arr TYPE int4_table,
          wa_int   TYPE i.

    IF k = 0.
      CLEAR result.
      RETURN.
    ENDIF.

    temp_arr = arr.
    SORT temp_arr.

    LOOP AT temp_arr INTO wa_int FROM ( lines( temp_arr ) - k + 1 ).
      APPEND wa_int TO result.
    ENDLOOP.

  ENDMETHOD.

ENDCLASS.