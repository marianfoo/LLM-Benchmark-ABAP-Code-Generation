CLASS z_humaneval_108 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_int_table,
             value TYPE i,
           END OF ty_int_table.
    TYPES: tt_int_table TYPE TABLE OF ty_int_table.

    CLASS-METHODS: count_nums
      IMPORTING
        VALUE(arr)    TYPE tt_int_table
      RETURNING
        VALUE(result) TYPE i.

  PRIVATE SECTION.
    CLASS-METHODS: digits_sum
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_108 IMPLEMENTATION.

  METHOD count_nums.
    DATA: lv_count TYPE i VALUE 0,
          lv_sum   TYPE i.

    LOOP AT arr INTO DATA(wa_arr).
      lv_sum = digits_sum( wa_arr-value ).
      IF lv_sum > 0.
        lv_count = lv_count + 1.
      ENDIF.
    ENDLOOP.

    result = lv_count.
  ENDMETHOD.

  METHOD digits_sum.
    DATA: lv_n      TYPE i,
          lv_neg    TYPE i VALUE 1,
          lv_string TYPE string,
          lv_char   TYPE c LENGTH 1,
          lv_digit  TYPE i,
          lv_sum    TYPE i VALUE 0,
          lv_first  TYPE abap_bool VALUE abap_true,
          lv_len    TYPE i,
          lv_index  TYPE i.

    lv_n = n.

    IF lv_n < 0.
      lv_n = lv_n * -1.
      lv_neg = -1.
    ENDIF.

    lv_string = |{ lv_n }|.
    lv_len = strlen( lv_string ).

    DO lv_len TIMES.
      lv_index = sy-index - 1.
      lv_char = lv_string+lv_index(1).
      lv_digit = lv_char.

      IF lv_first = abap_true.
        lv_digit = lv_digit * lv_neg.
        lv_first = abap_false.
      ENDIF.

      lv_sum = lv_sum + lv_digit.
    ENDDO.

    result = lv_sum.
  ENDMETHOD.

ENDCLASS.