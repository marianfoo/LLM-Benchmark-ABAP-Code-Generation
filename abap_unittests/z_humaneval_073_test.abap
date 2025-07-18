CLASS ltcl_test_smallest_change DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_smallest_change IMPLEMENTATION.
  METHOD test_case_1.
    DATA: arr    TYPE TABLE OF i,
          result TYPE i.
    arr = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 5 ) ( 4 ) ( 7 ) ( 9 ) ( 6 ) ).
    result = z_humaneval_073=>smallest_change( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 4 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: arr    TYPE TABLE OF i,
          result TYPE i.
    arr = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 3 ) ( 2 ) ( 2 ) ).
    result = z_humaneval_073=>smallest_change( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: arr    TYPE TABLE OF i,
          result TYPE i.
    arr = VALUE #( ( 1 ) ( 4 ) ( 2 ) ).
    result = z_humaneval_073=>smallest_change( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: arr    TYPE TABLE OF i,
          result TYPE i.
    arr = VALUE #( ( 1 ) ( 4 ) ( 4 ) ( 2 ) ).
    result = z_humaneval_073=>smallest_change( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: arr    TYPE TABLE OF i,
          result TYPE i.
    arr = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 2 ) ( 1 ) ).
    result = z_humaneval_073=>smallest_change( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: arr    TYPE TABLE OF i,
          result TYPE i.
    arr = VALUE #( ( 3 ) ( 1 ) ( 1 ) ( 3 ) ).
    result = z_humaneval_073=>smallest_change( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: arr    TYPE TABLE OF i,
          result TYPE i.
    arr = VALUE #( ( 1 ) ).
    result = z_humaneval_073=>smallest_change( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: arr    TYPE TABLE OF i,
          result TYPE i.
    arr = VALUE #( ( 0 ) ( 1 ) ).
    result = z_humaneval_073=>smallest_change( arr ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.
ENDCLASS.