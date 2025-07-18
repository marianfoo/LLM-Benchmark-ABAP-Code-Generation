CLASS z_humaneval_155 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_result,
             even_count TYPE i,
             odd_count  TYPE i,
           END OF ty_result.

    CLASS-METHODS: even_odd_count
      IMPORTING
        num           TYPE i
      RETURNING
        VALUE(result) TYPE ty_result.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_155 IMPLEMENTATION.
  METHOD even_odd_count.
    DATA: lv_abs_num   TYPE string,
          lv_digit     TYPE c LENGTH 1,
          lv_digit_int TYPE i.

    lv_abs_num = |{ abs( num ) }|.

    DO strlen( lv_abs_num ) TIMES.
      lv_digit = lv_abs_num+0(1).
      lv_abs_num = lv_abs_num+1.
      lv_digit_int = lv_digit.

      IF lv_digit_int MOD 2 = 0.
        result-even_count = result-even_count + 1.
      ELSE.
        result-odd_count = result-odd_count + 1.
      ENDIF.
    ENDDO.
  ENDMETHOD.
ENDCLASS.