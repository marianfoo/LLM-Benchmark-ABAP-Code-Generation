CLASS z_humaneval_147 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: get_max_triples
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE i.
ENDCLASS.

CLASS z_humaneval_147 IMPLEMENTATION.
  METHOD get_max_triples.
    DATA: a     TYPE int4_table,
          count TYPE i VALUE 0,
          i     TYPE i,
          j     TYPE i,
          k     TYPE i,
          val   TYPE i.

    DO n TIMES.
      i = sy-index.
      val = i * i - i + 1.
      APPEND val TO a.
    ENDDO.

    DO n TIMES.
      i = sy-index - 1.
      j = i + 1.
      WHILE j < n.
        k = j + 1.
        WHILE k < n.
          READ TABLE a INTO DATA(a_i) INDEX i + 1.
          READ TABLE a INTO DATA(a_j) INDEX j + 1.
          READ TABLE a INTO DATA(a_k) INDEX k + 1.
          IF ( a_i + a_j + a_k ) MOD 3 = 0.
            count = count + 1.
          ENDIF.
          k = k + 1.
        ENDWHILE.
        j = j + 1.
      ENDWHILE.
    ENDDO.

    result = count.
  ENDMETHOD.
ENDCLASS.