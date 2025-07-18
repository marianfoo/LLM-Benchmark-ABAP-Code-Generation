CLASS z_humaneval_082 DEFINITION PUBLIC.
  PUBLIC SECTION.
    CLASS-METHODS:
      prime_length
        IMPORTING iv_string                 TYPE string
        RETURNING VALUE(rv_is_prime_length) TYPE abap_bool.
ENDCLASS.

CLASS z_humaneval_082 IMPLEMENTATION.
  METHOD prime_length.
    DATA(lv_length) = strlen( iv_string ).

    " Handle edge cases
    IF lv_length < 2.
      rv_is_prime_length = abap_false.
      RETURN.
    ENDIF.

    DO.
      DATA(lv_divisor) = sy-index + 1.
      IF lv_divisor * lv_divisor > lv_length.
        EXIT.
      ENDIF.

      IF lv_length MOD lv_divisor = 0.
        rv_is_prime_length = abap_false.
        RETURN.
      ENDIF.
    ENDDO.

    rv_is_prime_length = abap_true.
  ENDMETHOD.
ENDCLASS.
