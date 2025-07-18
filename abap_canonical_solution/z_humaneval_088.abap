CLASS z_humaneval_088 DEFINITION PUBLIC FINAL CREATE PUBLIC.
  PUBLIC SECTION.
    CLASS-METHODS sort_array
      IMPORTING iv_array         TYPE int4_table
      RETURNING VALUE(rv_result) TYPE int4_table.
ENDCLASS.

CLASS z_humaneval_088 IMPLEMENTATION.
  METHOD sort_array.
    DATA lv_len TYPE i.
    lv_len = lines( iv_array ).
    IF lv_len = 0.
      rv_result = VALUE int4_table( ).
      RETURN.
    ENDIF.

    DATA lv_sum TYPE i.
    lv_sum = iv_array[ 1 ] + iv_array[ lv_len ].

    rv_result = iv_array.
    SORT rv_result ASCENDING.
    IF ( lv_sum MOD 2 ) = 0.
      SORT rv_result DESCENDING.
    ENDIF.
  ENDMETHOD.
ENDCLASS.
