CLASS ltcl_test_numerical_letter DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_grade,
             gpa TYPE f,
           END OF ty_grade.
    TYPES: tt_grades TYPE TABLE OF ty_grade.

    METHODS: test_mixed_grades FOR TESTING,
      test_single_d_plus FOR TESTING,
      test_single_d_minus FOR TESTING,
      test_single_e FOR TESTING,
      test_various_grades FOR TESTING,
      test_zero_and_d_minus FOR TESTING.

ENDCLASS.

CLASS ltcl_test_numerical_letter IMPLEMENTATION.

  METHOD test_mixed_grades.
    DATA: lt_grades   TYPE tt_grades,
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    lt_grades = VALUE #( ( gpa = '4.0' ) ( gpa = '3' ) ( gpa = '1.7' ) ( gpa = '2' ) ( gpa = '3.5' ) ).
    lt_expected = VALUE #( ( |A+| ) ( |B| ) ( |C-| ) ( |C| ) ( |A-| ) ).

    lt_actual = z_humaneval_081=>numerical_letter_grade( lt_grades ).

    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_single_d_plus.
    DATA: lt_grades   TYPE tt_grades,
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    lt_grades = VALUE #( ( gpa = '1.2' ) ).
    lt_expected = VALUE #( ( |D+| ) ).

    lt_actual = z_humaneval_081=>numerical_letter_grade( lt_grades ).

    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_single_d_minus.
    DATA: lt_grades   TYPE tt_grades,
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    lt_grades = VALUE #( ( gpa = '0.5' ) ).
    lt_expected = VALUE #( ( |D-| ) ).

    lt_actual = z_humaneval_081=>numerical_letter_grade( lt_grades ).

    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_single_e.
    DATA: lt_grades   TYPE tt_grades,
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    lt_grades = VALUE #( ( gpa = '0.0' ) ).
    lt_expected = VALUE #( ( |E| ) ).

    lt_actual = z_humaneval_081=>numerical_letter_grade( lt_grades ).

    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_various_grades.
    DATA: lt_grades   TYPE tt_grades,
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    lt_grades = VALUE #( ( gpa = '1' ) ( gpa = '0.3' ) ( gpa = '1.5' ) ( gpa = '2.8' ) ( gpa = '3.3' ) ).
    lt_expected = VALUE #( ( |D| ) ( |D-| ) ( |C-| ) ( |B| ) ( |B+| ) ).

    lt_actual = z_humaneval_081=>numerical_letter_grade( lt_grades ).

    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

  METHOD test_zero_and_d_minus.
    DATA: lt_grades   TYPE tt_grades,
          lt_expected TYPE string_table,
          lt_actual   TYPE string_table.

    lt_grades = VALUE #( ( gpa = '0' ) ( gpa = '0.7' ) ).
    lt_expected = VALUE #( ( |E| ) ( |D-| ) ).

    lt_actual = z_humaneval_081=>numerical_letter_grade( lt_grades ).

    cl_abap_unit_assert=>assert_equals( act = lt_actual exp = lt_expected ).
  ENDMETHOD.

ENDCLASS.