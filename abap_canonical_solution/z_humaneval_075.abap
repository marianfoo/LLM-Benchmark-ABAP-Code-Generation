CLASS z_humaneval_075 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: is_multiply_prime
      IMPORTING
        VALUE(a)      TYPE i
      RETURNING
        VALUE(result) TYPE abap_bool.

  PRIVATE SECTION.
    CLASS-METHODS: is_prime
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_075 IMPLEMENTATION.

  METHOD is_prime.
    DATA: j TYPE i.

    IF n < 2.
      result = abap_false.
      RETURN.
    ENDIF.

    j = 2.
    WHILE j < n.
      IF n MOD j = 0.
        result = abap_false.
        RETURN.
      ENDIF.
      j = j + 1.
    ENDWHILE.

    result = abap_true.
  ENDMETHOD.

  METHOD is_multiply_prime.
    DATA: i TYPE i,
          j TYPE i,
          k TYPE i.

    i = 2.
    WHILE i <= 100.
      IF is_prime( i ) = abap_true.
        j = 2.
        WHILE j <= 100.
          IF is_prime( j ) = abap_true.
            k = 2.
            WHILE k <= 100.
              IF is_prime( k ) = abap_true.
                IF i * j * k = a.
                  result = abap_true.
                  RETURN.
                ENDIF.
              ENDIF.
              k = k + 1.
            ENDWHILE.
          ENDIF.
          j = j + 1.
        ENDWHILE.
      ENDIF.
      i = i + 1.
    ENDWHILE.

    result = abap_false.
  ENDMETHOD.

ENDCLASS.