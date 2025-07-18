CLASS z_humaneval_059 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS largest_prime_factor
      IMPORTING
        n             TYPE i
      RETURNING
        VALUE(result) TYPE i.

  PRIVATE SECTION.
    CLASS-METHODS is_prime
      IMPORTING
        k             TYPE i
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_059 IMPLEMENTATION.

  METHOD largest_prime_factor.
    DATA: largest TYPE i VALUE 1,
          j       TYPE i.

    DO.
      j = sy-index + 1.
      IF j > n.
        EXIT.
      ENDIF.
      IF n MOD j = 0 AND is_prime( j ) = abap_true.
        IF j > largest.
          largest = j.
        ENDIF.
      ENDIF.
    ENDDO.

    result = largest.
  ENDMETHOD.

  METHOD is_prime.
    DATA: i TYPE i.

    IF k < 2.
      result = abap_false.
      RETURN.
    ENDIF.

    DO.
      i = sy-index + 1.
      IF i >= k.
        EXIT.
      ENDIF.
      IF k MOD i = 0.
        result = abap_false.
        RETURN.
      ENDIF.
    ENDDO.

    result = abap_true.
  ENDMETHOD.

ENDCLASS.