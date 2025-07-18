CLASS z_humaneval_096 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS: count_up_to
      IMPORTING iv_n             TYPE i
      RETURNING VALUE(rt_primes) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_096 IMPLEMENTATION.

  METHOD count_up_to.
    DATA: lv_i        TYPE i,
          lv_j        TYPE i,
          lv_is_prime TYPE abap_bool.

    CLEAR rt_primes.

    DO.
      lv_i = sy-index + 1.
      IF lv_i >= iv_n.
        EXIT.
      ENDIF.

      lv_is_prime = abap_true.

      lv_j = 2.
      WHILE lv_j < lv_i.
        IF lv_i MOD lv_j = 0.
          lv_is_prime = abap_false.
          EXIT.
        ENDIF.
        lv_j = lv_j + 1.
      ENDWHILE.

      IF lv_is_prime = abap_true.
        APPEND lv_i TO rt_primes.
      ENDIF.
    ENDDO.

  ENDMETHOD.

ENDCLASS.