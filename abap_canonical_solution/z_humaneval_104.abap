CLASS z_humaneval_104 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_int_table,
             value TYPE i,
           END OF ty_int_table,
           tt_int_table TYPE STANDARD TABLE OF ty_int_table WITH NON-UNIQUE DEFAULT KEY.

    CLASS-METHODS: unique_digits
      IMPORTING
        VALUE(x)      TYPE tt_int_table
      RETURNING
        VALUE(result) TYPE tt_int_table.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_104 IMPLEMENTATION.

  METHOD unique_digits.
    DATA: lv_num_str   TYPE string,
          lv_digit     TYPE c LENGTH 1,
          lv_digit_int TYPE i,
          lv_has_even  TYPE abap_bool,
          lv_offset    TYPE i,
          wa_result    TYPE ty_int_table,
          wa_x         TYPE ty_int_table.

    LOOP AT x INTO wa_x.
      lv_num_str = |{ wa_x-value }|.
      lv_has_even = abap_false.

      DO strlen( lv_num_str ) TIMES.
        lv_offset = sy-index - 1.
        lv_digit = lv_num_str+lv_offset(1).
        lv_digit_int = lv_digit - '0'.
        IF lv_digit_int MOD 2 = 0.
          lv_has_even = abap_true.
          EXIT.
        ENDIF.
      ENDDO.

      IF lv_has_even = abap_false.
        wa_result-value = wa_x-value.
        APPEND wa_result TO result.
      ENDIF.
    ENDLOOP.

    SORT result BY value ASCENDING.

  ENDMETHOD.

ENDCLASS.