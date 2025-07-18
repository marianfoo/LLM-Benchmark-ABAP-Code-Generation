CLASS z_humaneval_131 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: digits
      IMPORTING
        n             TYPE i
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_131 IMPLEMENTATION.

  METHOD digits.
    DATA: n_string   TYPE string,
          digit_char TYPE c LENGTH 1,
          digit_int  TYPE i,
          product    TYPE i VALUE 1,
          odd_count  TYPE i VALUE 0,
          index      TYPE i.

    n_string = |{ n }|.

    DO strlen( n_string ) TIMES.
      index = sy-index - 1.
      digit_char = n_string+index(1).
      digit_int = digit_char.

      IF digit_int MOD 2 = 1.
        product = product * digit_int.
        odd_count = odd_count + 1.
      ENDIF.
    ENDDO.

    IF odd_count = 0.
      result = 0.
    ELSE.
      result = product.
    ENDIF.

  ENDMETHOD.

ENDCLASS.