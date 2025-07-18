CLASS z_humaneval_094 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: skjkasdkd
      IMPORTING
        lst           TYPE int4_table
      RETURNING
        VALUE(result) TYPE i.

  PRIVATE SECTION.
    CLASS-METHODS: is_prime
      IMPORTING
        n            TYPE i
      RETURNING
        VALUE(prime) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_094 IMPLEMENTATION.

  METHOD skjkasdkd.
    DATA: maxx          TYPE i VALUE 0,
          i             TYPE i VALUE 0,
          len           TYPE i,
          current_value TYPE i,
          str_value     TYPE string,
          char_value    TYPE c LENGTH 1,
          digit_sum     TYPE i VALUE 0,
          digit_value   TYPE i,
          offset        TYPE i.

    len = lines( lst ).

    WHILE i < len.
      i = i + 1.
      READ TABLE lst INDEX i INTO current_value.
      IF current_value > maxx AND is_prime( current_value ).
        maxx = current_value.
      ENDIF.
    ENDWHILE.

    str_value = |{ maxx }|.

    DO strlen( str_value ) TIMES.
      offset = sy-index - 1.
      char_value = str_value+offset(1).
      digit_value = char_value.
      digit_sum = digit_sum + digit_value.
    ENDDO.

    result = digit_sum.
  ENDMETHOD.

  METHOD is_prime.
    DATA: i      TYPE i VALUE 2,
          sqrt_n TYPE f,
          max_i  TYPE i.

    IF n < 2.
      prime = abap_false.
      RETURN.
    ENDIF.

    sqrt_n = sqrt( n ).
    max_i = trunc( sqrt_n ) + 1.

    WHILE i < max_i.
      IF n MOD i = 0.
        prime = abap_false.
        RETURN.
      ENDIF.
      i = i + 1.
    ENDWHILE.

    prime = abap_true.
  ENDMETHOD.

ENDCLASS.