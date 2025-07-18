CLASS z_humaneval_103 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: rounded_avg
      IMPORTING
        n             TYPE i
        m             TYPE i
      RETURNING
        VALUE(result) TYPE string.

ENDCLASS.

CLASS z_humaneval_103 IMPLEMENTATION.
  METHOD rounded_avg.
    DATA: summation   TYPE i,
          avg         TYPE f,
          rounded_avg TYPE i,
          binary_str  TYPE string,
          temp        TYPE i,
          remainder   TYPE i.

    IF m < n.
      result = '-1'.
      RETURN.
    ENDIF.

    summation = 0.
    DO ( m - n + 1 ) TIMES.
      summation = summation + ( n + sy-index - 1 ).
    ENDDO.

    avg = summation / ( m - n + 1 ).

    DATA: fractional_part TYPE f.
    fractional_part = avg - trunc( avg ).

    IF fractional_part = '0.5'.
      IF trunc( avg ) MOD 2 = 0.
        rounded_avg = trunc( avg ).
      ELSE.
        rounded_avg = trunc( avg ) + 1.
      ENDIF.
    ELSE.
      rounded_avg = round( val = avg dec = 0 ).
    ENDIF.

    IF rounded_avg = 0.
      result = '0b0'.
      RETURN.
    ENDIF.

    temp = rounded_avg.
    WHILE temp > 0.
      remainder = temp MOD 2.
      IF remainder = 1.
        CONCATENATE '1' binary_str INTO binary_str.
      ELSE.
        CONCATENATE '0' binary_str INTO binary_str.
      ENDIF.
      temp = temp DIV 2.
    ENDWHILE.

    CONCATENATE '0b' binary_str INTO result.
  ENDMETHOD.
ENDCLASS.